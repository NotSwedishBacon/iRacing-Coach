"""Unit tests for the FastMCP server (`mcp_server/server.py`), RAG store (`rag/vector_store.py`), and Garage61 client (`telemetry/g61_client.py`)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from mcp_server import server
from mcp_server.server import analyze_last_session, mcp, query_setup_guide
from rag.vector_store import query_setup_advice
from scraper.ingest_rag import VALID_COMPONENTS, VALID_PHASES, VALID_SYMPTOMS, load_and_chunk_directory
from telemetry.g61_client import Garage61Client
from tests.test_math import make_reference_corner_lap


def test_rag_chunking_and_metadata_tags() -> None:
    """Verify all GitBook markdown chunks are split on ##/### headers and tagged with valid metadata."""
    chunks = load_and_chunk_directory()
    assert len(chunks) > 5
    for chunk in chunks:
        meta = chunk["metadata"]
        assert meta["phase"] in VALID_PHASES
        assert meta["symptom"] in VALID_SYMPTOMS
        assert meta["component"] in VALID_COMPONENTS


def test_query_setup_guide_tool() -> None:
    """Verify `query_setup_guide` returns relevant setup recommendations from the vector store."""
    results = query_setup_guide(corner_phase="entry", symptom="understeer", component="arb")
    assert isinstance(results, list)
    assert len(results) >= 1
    assert any("arb" in r["text"].lower() or "anti-roll" in r["text"].lower() for r in results)

    direct = query_setup_advice("front anti-roll bar entry understeer", phase="entry", symptom="understeer", top_k=2)
    assert len(direct) == 2


def test_analyze_last_session_holds_setup_when_driver_fault_isolated() -> None:
    """Verify `analyze_last_session` sets `allow_setup_adjustments: False` when driver faults exist."""
    ref_dict = make_reference_corner_lap()
    user_dict = make_reference_corner_lap()
    # Inject lazy brake hit into user lap
    s = user_dict["distance_m"]
    user_dict["brake"] *= 0.0
    user_dict["brake"][(s >= 45.0) & (s <= 68.0)] = 0.85 * ((s[(s >= 45.0) & (s <= 68.0)] - 45.0) / 23.0)
    user_dict["brake"][(s > 68.0) & (s <= 150.0)] = 0.50

    mock_client = MagicMock(spec=Garage61Client)
    mock_client.get_my_latest_lap.return_value = {
        "id": "lap_user_101",
        "car_slug": "porsche-992-gt3-r",
        "track_slug": "unmapped-test-track",
    }
    mock_client.get_fastest_comparison_lap.return_value = {
        "id": "lap_ref_999",
    }
    mock_client.download_lap_csv.side_effect = [
        pd.DataFrame(user_dict),
        pd.DataFrame(ref_dict),
    ]

    with patch("mcp_server.server.Garage61Client", return_value=mock_client):
        result = analyze_last_session(
            car="porsche-992-gt3-r",
            track="unmapped-test-track",
            specific_corner=1,
        )

    assert result["technique_fault_isolated"] is True
    assert result["allow_setup_adjustments"] is False
    assert result["setup_recommendations"] == []
    assert len(result["flaws"]) >= 1
    assert "HOLD SETUP CHANGES" in result["coaching_diagnostics"]


def test_analyze_last_session_unlocks_setup_when_driving_is_clean() -> None:
    """Verify `analyze_last_session` sets `allow_setup_adjustments: True` and queries RAG when driving is clean."""
    ref_dict = make_reference_corner_lap()
    user_dict = make_reference_corner_lap()
    # Clean inputs, slightly lower apex speed (vehicle balance limited)
    user_dict["speed"] = user_dict["speed"] - 0.5

    mock_client = MagicMock(spec=Garage61Client)
    mock_client.get_my_latest_lap.return_value = {
        "id": "lap_user_clean",
        "car_slug": "ferrari-296-gt3",
        "track_slug": "unmapped-test-track",
    }
    mock_client.get_fastest_comparison_lap.return_value = {
        "id": "lap_ref_clean",
    }
    mock_client.download_lap_csv.side_effect = [
        pd.DataFrame(user_dict),
        pd.DataFrame(ref_dict),
    ]

    with patch("mcp_server.server.Garage61Client", return_value=mock_client):
        result = analyze_last_session(
            car="ferrari-296-gt3",
            track="unmapped-test-track",
            specific_corner=1,
            symptom="understeer",
            corner_phase="entry",
        )

    assert result["technique_fault_isolated"] is False
    assert result["allow_setup_adjustments"] is True
    assert len(result["setup_recommendations"]) >= 1
    assert "INPUTS VERIFIED CLEAN" in result["coaching_diagnostics"]


def test_garage61_client_endpoints_and_fallback() -> None:
    """Verify Garage61Client calls `/me/laps`, falls back from team laps to public laps, and parses CSV."""
    mock_session = MagicMock()

    # 1. /me/laps response
    resp_me = MagicMock()
    resp_me.json.return_value = {"items": [{"id": "my_lap_1", "car": "gt3", "track": "spa"}]}

    # 2. /laps?team=... empty response -> triggers fallback to public clean lap
    resp_team = MagicMock()
    resp_team.ok = True
    resp_team.json.return_value = {"items": []}

    # 3. /laps?clean=true public response
    resp_pub = MagicMock()
    resp_pub.json.return_value = {"items": [{"id": "pub_lap_fast", "lap_time": 136.4}]}

    # 4. /laps/{id}/csv response
    resp_csv = MagicMock()
    resp_csv.text = "LapDist,Speed,Brake,Throttle,SteeringWheelAngle,LatAccel,LongAccel\n0,50,0,100,0,0,0\n10,52,0,100,0.1,4.9,1.0\n"

    mock_session.get.side_effect = [resp_me, resp_team, resp_pub, resp_csv]

    client = Garage61Client(api_key="test_token_123", session=mock_session)
    my_lap = client.get_my_latest_lap(car_slug="gt3", track_slug="spa")
    assert my_lap["id"] == "my_lap_1"

    comp_lap = client.get_fastest_comparison_lap(car_slug="gt3", track_slug="spa", team_id="team_42")
    assert comp_lap["id"] == "pub_lap_fast"

    df = client.download_lap_csv("pub_lap_fast")
    assert "distance_m" in df.columns
    assert "steering_deg" in df.columns
    assert df["throttle"].max() <= 1.0


def test_fastmcp_server_name() -> None:
    """Verify the FastMCP server is initialized with the required name."""
    assert mcp.name == "iracing-setup-engineer"


def test_all_43_garage61_openapi_endpoints_and_mcp_tools() -> None:
    """Verify every one of the 43 Garage 61 OpenAPI v1 operations is implemented and exposed as an MCP tool."""
    mock_session = MagicMock()

    def make_json_resp(status_code: int = 200, payload: dict | None = None, text: str = '{"ok": true}') -> MagicMock:
        r = MagicMock()
        r.status_code = status_code
        r.text = text
        r.content = b"\x00\x01\x02\x03"
        r.json.return_value = payload if payload is not None else {"ok": True}
        return r

    mock_session.get.return_value = make_json_resp(200, {"items": []}, "col1,col2\n1,2\n")
    mock_session.post.return_value = make_json_resp(201, {"created": True}, "")
    mock_session.delete.return_value = make_json_resp(204, None, "")

    client = Garage61Client(api_key="unit_test_key", session=mock_session)

    with patch("mcp_server.server.Garage61Client", return_value=client):
        # 1. GET /analyses
        assert server.g61_find_analyses(types=["lap", "sector"], trainingplan="tp1", limit=10, offset=0) == {"items": []}
        # 2. GET /analyses/{id}
        assert server.g61_get_analysis(id="an_1") == {"items": []}
        # 3. GET /car-groups
        assert server.g61_find_car_groups() == {"items": []}
        # 4. GET /cars
        assert server.g61_find_cars() == {"items": []}
        # 5. GET /events
        assert server.g61_find_events(platform="iracing", car=153, track=498, limit=5) == {"items": []}
        # 6. GET /events/{id}
        assert server.g61_get_event(id="ev_1") == {"items": []}
        # 7. GET /laps
        assert server.g61_find_laps(
            tracks=[498],
            cars=[153, -4],
            drivers=["me"],
            session_types=[1, 3],
            conditions_wind_dir=[1, 2],
            group="driver",
            limit=50,
        ) == {"items": []}
        # 8. GET /laps/{id}
        assert server.g61_get_lap(id="lap_1") == {"items": []}
        # 9. GET /laps/{id}/csv
        assert "col1,col2" in server.g61_get_lap_csv(id="lap_1")
        # 10. GET /laps/{id}/ghost.bin
        ghost = server.g61_get_lap_ghost(id="lap_1")
        assert ghost["lap_id"] == "lap_1"
        assert ghost["size_bytes"] == 4
        # 11. GET /me
        assert server.g61_get_me() == {"items": []}
        # 12. GET /me/accounts
        assert server.g61_get_accounts(rating_history=True) == {"items": []}
        # 13. POST /me/datapackgroups
        assert server.g61_create_user_data_pack_group(id="dpg_1")["status_code"] == 201
        # 14. DELETE /me/datapackgroups/{id}
        assert server.g61_delete_user_data_pack_group(id="dpg_1")["status_code"] == 204
        # 15. POST /me/datapacks
        assert server.g61_create_user_data_pack(id="dp_1")["status_code"] == 201
        # 16. DELETE /me/datapacks/{id}
        assert server.g61_delete_user_data_pack(id="dp_1")["status_code"] == 204
        # 17. GET /me/statistics
        assert server.g61_get_statistics(cars=[153], tracks=[498]) == {"items": []}
        # 18. GET /platforms
        assert server.g61_find_platforms() == {"items": []}
        # 19. GET /teams
        assert server.g61_find_teams() == {"items": []}
        # 20. GET /teams/{id}
        assert server.g61_get_team(id="team_1") == {"items": []}
        # 21. GET /teams/{team}/datapackgroups
        assert server.g61_get_team_data_pack_groups(team="team_1", limit=10) == {"items": []}
        # 22. GET /teams/{team}/datapackgroups/{id}
        assert server.g61_get_team_data_pack_group(team="team_1", id="dpg_1") == {"items": []}
        # 23. POST /teams/{team}/datapackgroups/{id}/members
        assert server.g61_create_data_pack_group_member(team="team_1", id="dpg_1", data_pack_id="dp_1")["status_code"] == 201
        # 24. DELETE /teams/{team}/datapackgroups/{id}/members/{member}
        assert server.g61_delete_data_pack_group_member(team="team_1", id="dpg_1", member="dp_1")["status_code"] == 204
        # 25. DELETE /teams/{team}/datapackgroups/{id}/subscribers/{subscriber}
        assert server.g61_delete_data_pack_group_subscriber(team="team_1", id="dpg_1", subscriber="usr_1")["status_code"] == 204
        # 26. GET /teams/{team}/datapacks
        assert server.g61_get_team_data_packs(team="team_1", limit=10) == {"items": []}
        # 27. GET /teams/{team}/datapacks/{id}
        assert server.g61_get_team_data_pack(team="team_1", id="dp_1") == {"items": []}
        # 28. POST /teams/{team}/datapacks/{id}
        assert server.g61_update_team_data_pack(team="team_1", id="dp_1", name="Week 1", archived=False)["status_code"] == 201
        # 29. POST /teams/{team}/datapacks/{id}/content
        assert server.g61_create_team_data_pack_content(team="team_1", id="dp_1", video_url="https://youtu.be/xyz")["status_code"] == 201
        # 30. POST /teams/{team}/datapacks/{id}/content/{item}
        assert server.g61_update_team_data_pack_content(team="team_1", id="dp_1", item="item_1", video_url="https://youtu.be/abc")["status_code"] == 201
        # 31. DELETE /teams/{team}/datapacks/{id}/content/{item}
        assert server.g61_delete_team_data_pack_content(team="team_1", id="dp_1", item="item_1")["status_code"] == 204
        # 32. GET /teams/{team}/datapacks/{id}/content/{item}/ghost.bin
        dp_ghost = server.g61_get_team_data_pack_content_ghost_lap_bin(team="team_1", id="dp_1", item="item_1")
        assert dp_ghost["size_bytes"] == 4
        # 33. GET /teams/{team}/datapacks/{id}/content/{item}/lap.csv
        assert "col1,col2" in server.g61_get_team_data_pack_content_lap_csv(team="team_1", id="dp_1", item="item_1")
        # 34. POST /teams/{team}/datapacks/{id}/content/{item}/move
        assert server.g61_move_team_data_pack_content(team="team_1", id="dp_1", item="item_1", position=1)["status_code"] == 201
        # 35. GET /teams/{team}/datapacks/{id}/content/{item}/replay.bin
        dp_replay = server.g61_get_team_data_pack_content_replay_bin(team="team_1", id="dp_1", item="item_1")
        assert dp_replay["size_bytes"] == 4
        # 36. GET /teams/{team}/datapacks/{id}/content/{item}/setup.sto
        dp_setup = server.g61_get_team_data_pack_content_setup_sto(team="team_1", id="dp_1", item="item_1")
        assert dp_setup["size_bytes"] == 4
        # 37. DELETE /teams/{team}/datapacks/{id}/subscribers/{subscriber}
        assert server.g61_delete_data_pack_subscriber(team="team_1", id="dp_1", subscriber="usr_1")["status_code"] == 204
        # 38. POST /teams/{team}/invites
        assert server.g61_create_team_invite(team="team_1", name="New Driver Invite", max_usages=5, expires_after=3600)["status_code"] == 201
        # 39. DELETE /teams/{team}/members/{id}
        assert server.g61_delete_team_member(team="team_1", id="usr_1")["status_code"] == 204
        # 40. GET /teams/{team}/statistics
        assert server.g61_get_team_statistics(team="team_1", cars=[153], tracks=[498]) == {"items": []}
        # 41. GET /teams/{team}/trainingplans
        assert server.g61_get_team_training_plans(team="team_1", limit=10) == {"items": []}
        # 42. GET /teams/{team}/trainingplans/{id}
        assert server.g61_get_team_training_plan(team="team_1", id="tp_1") == {"items": []}
        # 43. GET /tracks
        assert server.g61_find_tracks() == {"items": []}

    # Verify total calls across all 43 endpoints:
    # 28 GET calls + 8 POST calls + 7 DELETE calls = 43 calls
    assert mock_session.get.call_count == 28
    assert mock_session.post.call_count == 8
    assert mock_session.delete.call_count == 7


