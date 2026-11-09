"""Unit tests for the FastMCP server (`mcp_server/server.py`), RAG store (`rag/vector_store.py`), and Garage61 client (`telemetry/g61_client.py`)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

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
