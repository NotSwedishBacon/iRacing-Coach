"""Unified FastMCP Server exposing telemetry physics math and GitBook setup RAG tools.

Exposes `FastMCP("iracing-setup-engineer")` with:
1. `analyze_last_session(car: str = None, track: str = None, specific_corner: int = None) -> dict`
   - Automates lap resolution, 60 Hz CSV download, 1.0m spatial alignment, and mathematical analysis.
   - Decision Gate:
     * If driving faults are found (`technique_fault_isolated == True`), sets
       `allow_setup_adjustments: False` and returns only coaching diagnostics.
     * If driving is clean (`technique_fault_isolated == False`), queries the GitBook RAG
       and attaches setup adjustments (`allow_setup_adjustments: True`).
2. `query_setup_guide(corner_phase: str, symptom: str, component: str = None) -> list`
   - Explicit RAG lookup tool for setup guidance.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP  # type: ignore[no-redef]

from mcp_server.persona_prompts import RACE_ENGINEER_SYSTEM_PROMPT, format_coaching_brief
from rag.vector_store import query_setup_advice
from telemetry.g61_client import Garage61Client
from telemetry.signal_math import analyze_telemetry

logger = logging.getLogger(__name__)

mcp = FastMCP("iracing-setup-engineer")


def _infer_clean_lap_balance_symptom(analysis: dict[str, Any]) -> tuple[str, str]:
    """Infer target (phase, symptom) for setup tuning when driver inputs are verified clean."""
    corners = analysis.get("corners") or []
    if not corners:
        return ("mid", "understeer")

    # Identify corner with largest speed deficit relative to reference
    slowest_corner = min(corners, key=lambda c: float(c.get("delta_v_min", 0.0)))
    scrub = float(slowest_corner.get("steering_scrub_delta_deg", 0.0))
    us_delta = float(slowest_corner.get("understeer_index_delta", 0.0))
    exit_steer = float(slowest_corner.get("user_exit_steering_at_50pct_throttle_deg", 0.0))

    if scrub > 4.0 or us_delta > 0.15:
        return ("mid", "understeer")
    if exit_steer > 15.0 or us_delta < -0.15:
        return ("exit", "oversteer")
    return ("entry", "understeer")


@mcp.prompt(name="race_engineer_persona")
def get_race_engineer_persona() -> str:
    """Return the Elite Performance Driver Coach / Race Engineer system prompt."""
    return RACE_ENGINEER_SYSTEM_PROMPT


@mcp.tool()
def analyze_last_session(
    car: str | None = None,
    track: str | None = None,
    specific_corner: int | None = None,
    symptom: str | None = None,
    corner_phase: str | None = None,
) -> dict[str, Any]:
    """Analyze the user's latest Garage61 session against the fastest reference lap.

    Automates lap resolution, 60 Hz CSV telemetry download, 1.0m spatial distance
    alignment, and deterministic NumPy/SciPy analysis.

    Decision Gate:
    - If driver technique faults are isolated (`technique_fault_isolated == True`),
      sets `allow_setup_adjustments: False` and returns only coaching diagnostics.
    - If driving inputs are clean (`technique_fault_isolated == False`), queries the
      GitBook RAG and attaches targeted vehicle setup adjustments.
    """
    g61 = Garage61Client()

    user_lap_meta = g61.get_my_latest_lap(car_slug=car, track_slug=track)
    resolved_car = car or user_lap_meta.get("car_slug") or user_lap_meta.get("car") or "gt3"
    resolved_track = track or user_lap_meta.get("track_slug") or user_lap_meta.get("track") or "spa"
    team_id = user_lap_meta.get("team_id") or user_lap_meta.get("team")

    if isinstance(resolved_car, dict):
        resolved_car = resolved_car.get("slug") or resolved_car.get("id") or "gt3"
    if isinstance(resolved_track, dict):
        resolved_track = resolved_track.get("slug") or resolved_track.get("id") or "spa"

    ref_lap_meta = g61.get_fastest_comparison_lap(
        car_slug=str(resolved_car),
        track_slug=str(resolved_track),
        team_id=str(team_id) if team_id else None,
    )

    user_lap_id = str(user_lap_meta["id"])
    ref_lap_id = str(ref_lap_meta["id"])

    user_df = g61.download_lap_csv(user_lap_id)
    ref_df = g61.download_lap_csv(ref_lap_id)

    math_report = analyze_telemetry(
        user_lap=user_df,
        ref_lap=ref_df,
        track_slug=str(resolved_track),
        specific_corner=specific_corner,
    )

    technique_fault_isolated = bool(math_report["technique_fault_isolated"])
    allow_setup_adjustments = not technique_fault_isolated

    setup_recommendations: list[dict[str, Any]] = []
    if allow_setup_adjustments:
        inferred_phase, inferred_symptom = _infer_clean_lap_balance_symptom(math_report)
        target_phase = (corner_phase or inferred_phase).lower().strip()
        target_symptom = (symptom or inferred_symptom).lower().strip()
        query_text = f"Fix {target_Phase_label(target_phase)} {target_symptom} car setup adjustment"
        setup_recommendations = query_setup_advice(
            query=query_text,
            phase=target_phase,
            symptom=target_symptom,
            top_k=3,
        )

    payload: dict[str, Any] = {
        "car": str(resolved_car),
        "track": str(resolved_track),
        "specific_corner": specific_corner,
        "user_lap_id": user_lap_id,
        "reference_lap_id": ref_lap_id,
        "technique_fault_isolated": technique_fault_isolated,
        "allow_setup_adjustments": allow_setup_adjustments,
        "flaws": math_report["flaws"],
        "corners": math_report["corners"],
        "summary_deltas": math_report["summary_deltas"],
        "top_time_loss_corners": math_report.get("top_time_loss_corners", []),
        "setup_recommendations": setup_recommendations,
        "persona_system_prompt": RACE_ENGINEER_SYSTEM_PROMPT,
    }
    payload["coaching_diagnostics"] = format_coaching_brief(payload)
    return payload


def target_Phase_label(phase: str) -> str:
    if phase == "mid":
        return "mid-corner"
    if phase in ("entry", "exit"):
        return f"corner {phase}"
    return phase


@mcp.tool()
def query_setup_guide(
    corner_phase: str,
    symptom: str,
    component: str | None = None,
) -> list[dict[str, Any]]:
    """Explicit RAG lookup tool for iRacing vehicle setup guidance from the GitBook knowledge base.

    Args:
        corner_phase: Corner phase ("entry", "mid", "exit", or "general").
        symptom: Handling symptom ("understeer", "oversteer", "tire_wear", "bottoming", or "general").
        component: Optional component filter ("arb", "springs", "dampers_low", "dampers_fast",
                   "diff", "camber_toe", "brake_bias", "aero_rake").
    """
    comp_clause = f" using {component}" if component else ""
    query_str = f"How to fix {corner_phase} {symptom}{comp_clause} in iRacing car setup"
    return query_setup_advice(
        query=query_str,
        phase=corner_phase,
        symptom=symptom,
        component=component,
        top_k=3,
    )


# =============================================================================
# Complete Garage 61 API v1 MCP Tools (All 43 OpenAPI Endpoints)
# =============================================================================


# 1. GET /analyses (operationId: findAnalyses)
@mcp.tool()
def g61_find_analyses(
    types: list[str] | str | None = None,
    trainingplan: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any] | list[Any]:
    """Analyses for current user (`GET /analyses`, operationId: `findAnalyses`)."""
    return Garage61Client().find_analyses(
        types=types,
        trainingplan=trainingplan,
        limit=limit,
        offset=offset,
    )


# 2. GET /analyses/{id} (operationId: getAnalysis)
@mcp.tool()
def g61_get_analysis(id: str) -> dict[str, Any] | list[Any]:
    """Get analysis details (`GET /analyses/{id}`, operationId: `getAnalysis`)."""
    return Garage61Client().get_analysis(id=id)


# 3. GET /car-groups (operationId: findCarGroups)
@mcp.tool()
def g61_find_car_groups() -> dict[str, Any] | list[Any]:
    """Available car groups (`GET /car-groups`, operationId: `findCarGroups`)."""
    return Garage61Client().find_car_groups()


# 4. GET /cars (operationId: findCars)
@mcp.tool()
def g61_find_cars() -> dict[str, Any] | list[Any]:
    """Available cars (`GET /cars`, operationId: `findCars`)."""
    return Garage61Client().find_cars()


# 5. GET /events (operationId: findEvents)
@mcp.tool()
def g61_find_events(
    platform: str | None = None,
    platform_event: str | None = None,
    after: str | None = None,
    before: str | None = None,
    car: int | None = None,
    track: int | None = None,
    user: str | None = None,
    team: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any] | list[Any]:
    """Find events (`GET /events`, operationId: `findEvents`)."""
    return Garage61Client().find_events(
        platform=platform,
        platform_event=platform_event,
        after=after,
        before=before,
        car=car,
        track=track,
        user=user,
        team=team,
        limit=limit,
        offset=offset,
    )


# 6. GET /events/{id} (operationId: getEvent)
@mcp.tool()
def g61_get_event(id: str) -> dict[str, Any] | list[Any]:
    """Get event details (`GET /events/{id}`, operationId: `getEvent`)."""
    return Garage61Client().get_event(id=id)


# 7. GET /laps (operationId: findLaps)
@mcp.tool()
def g61_find_laps(
    tracks: list[int] | str | int | None = None,
    cars: list[int] | str | int | None = None,
    seasons: list[int] | str | int | None = None,
    drivers: list[str] | str | None = None,
    teams: list[str] | str | None = None,
    extra_drivers: list[str] | str | None = None,
    age: int | None = None,
    after: str | None = None,
    session_types: list[int] | str | int | None = None,
    session_setup_types: list[int] | str | int | None = None,
    see_telemetry: bool | None = None,
    see_ghost_lap: bool | None = None,
    see_setup: bool | None = None,
    lap_types: list[int] | str | int | None = None,
    push_to_pass: bool | None = None,
    event: str | None = None,
    unclean: bool | None = None,
    min_weight_penalty: float | None = None,
    max_weight_penalty: float | None = None,
    min_power_adjust: float | None = None,
    max_power_adjust: float | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    min_fuel: float | None = None,
    max_fuel: float | None = None,
    min_fuel_used: float | None = None,
    max_fuel_used: float | None = None,
    min_lap_time: float | None = None,
    max_lap_time: float | None = None,
    min_conditions_track_usage: int | None = None,
    max_conditions_track_usage: int | None = None,
    min_conditions_track_wetness: int | None = None,
    max_conditions_track_wetness: int | None = None,
    min_conditions_track_temp: float | None = None,
    max_conditions_track_temp: float | None = None,
    min_conditions_air_temp: float | None = None,
    max_conditions_air_temp: float | None = None,
    min_conditions_wind_vel: float | None = None,
    max_conditions_wind_vel: float | None = None,
    min_conditions_relative_humidity: float | None = None,
    max_conditions_relative_humidity: float | None = None,
    min_conditions_fog_level: float | None = None,
    max_conditions_fog_level: float | None = None,
    min_conditions_precipitation: float | None = None,
    max_conditions_precipitation: float | None = None,
    min_conditions_cloud: int | None = None,
    max_conditions_cloud: int | None = None,
    conditions_wind_dir: list[int] | str | int | None = None,
    round_units: str | None = None,
    group: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any] | list[Any]:
    """Find laps and lap records (`GET /laps`, operationId: `findLaps`)."""
    return Garage61Client().find_laps(
        tracks=tracks,
        cars=cars,
        seasons=seasons,
        drivers=drivers,
        teams=teams,
        extra_drivers=extra_drivers,
        age=age,
        after=after,
        session_types=session_types,
        session_setup_types=session_setup_types,
        see_telemetry=see_telemetry,
        see_ghost_lap=see_ghost_lap,
        see_setup=see_setup,
        lap_types=lap_types,
        push_to_pass=push_to_pass,
        event=event,
        unclean=unclean,
        min_weight_penalty=min_weight_penalty,
        max_weight_penalty=max_weight_penalty,
        min_power_adjust=min_power_adjust,
        max_power_adjust=max_power_adjust,
        min_rating=min_rating,
        max_rating=max_rating,
        min_fuel=min_fuel,
        max_fuel=max_fuel,
        min_fuel_used=min_fuel_used,
        max_fuel_used=max_fuel_used,
        min_lap_time=min_lap_time,
        max_lap_time=max_lap_time,
        min_conditions_track_usage=min_conditions_track_usage,
        max_conditions_track_usage=max_conditions_track_usage,
        min_conditions_track_wetness=min_conditions_track_wetness,
        max_conditions_track_wetness=max_conditions_track_wetness,
        min_conditions_track_temp=min_conditions_track_temp,
        max_conditions_track_temp=max_conditions_track_temp,
        min_conditions_air_temp=min_conditions_air_temp,
        max_conditions_air_temp=max_conditions_air_temp,
        min_conditions_wind_vel=min_conditions_wind_vel,
        max_conditions_wind_vel=max_conditions_wind_vel,
        min_conditions_relative_humidity=min_conditions_relative_humidity,
        max_conditions_relative_humidity=max_conditions_relative_humidity,
        min_conditions_fog_level=min_conditions_fog_level,
        max_conditions_fog_level=max_conditions_fog_level,
        min_conditions_precipitation=min_conditions_precipitation,
        max_conditions_precipitation=max_conditions_precipitation,
        min_conditions_cloud=min_conditions_cloud,
        max_conditions_cloud=max_conditions_cloud,
        conditions_wind_dir=conditions_wind_dir,
        round_units=round_units,
        group=group,
        limit=limit,
        offset=offset,
    )


# 8. GET /laps/{id} (operationId: getLap)
@mcp.tool()
def g61_get_lap(id: str) -> dict[str, Any] | list[Any]:
    """Get information about a specific lap (`GET /laps/{id}`, operationId: `getLap`)."""
    return Garage61Client().get_lap(id=id)


# 9. GET /laps/{id}/csv (operationId: getLapCSV)
@mcp.tool()
def g61_get_lap_csv(id: str) -> str:
    """Export the telemetry for a lap as a CSV file (`GET /laps/{id}/csv`, operationId: `getLapCSV`)."""
    return Garage61Client().get_lap_csv(id=id)


# 10. GET /laps/{id}/ghost.bin (operationId: getLapGhost)
@mcp.tool()
def g61_get_lap_ghost(id: str) -> dict[str, Any]:
    """Download the ghost lap file for a lap (`GET /laps/{id}/ghost.bin`, operationId: `getLapGhost`)."""
    raw = Garage61Client().get_lap_ghost(id=id)
    return {
        "lap_id": id,
        "content_type": "application/octet-stream",
        "size_bytes": len(raw),
        "data_base64": base64.b64encode(raw).decode("ascii"),
    }


# 11. GET /me (operationId: me)
@mcp.tool()
def g61_get_me() -> dict[str, Any] | list[Any]:
    """Information about the currently authenticated user (`GET /me`, operationId: `me`)."""
    return Garage61Client().get_me()


# 12. GET /me/accounts (operationId: getAccounts)
@mcp.tool()
def g61_get_accounts(rating_history: bool | None = None) -> dict[str, Any] | list[Any]:
    """Get linked accounts for the current user (`GET /me/accounts`, operationId: `getAccounts`)."""
    return Garage61Client().get_accounts(rating_history=rating_history)


# 13. POST /me/datapackgroups (operationId: createUserDataPackGroup)
@mcp.tool()
def g61_create_user_data_pack_group(id: str) -> dict[str, Any] | list[Any]:
    """Subscribe user to a data pack group (`POST /me/datapackgroups`, operationId: `createUserDataPackGroup`)."""
    return Garage61Client().create_user_data_pack_group(id=id)


# 14. DELETE /me/datapackgroups/{id} (operationId: deleteUserDataPackGroup)
@mcp.tool()
def g61_delete_user_data_pack_group(id: str) -> dict[str, Any] | list[Any]:
    """Unsubscribe user from a data pack group (`DELETE /me/datapackgroups/{id}`, operationId: `deleteUserDataPackGroup`)."""
    return Garage61Client().delete_user_data_pack_group(id=id)


# 15. POST /me/datapacks (operationId: createUserDataPack)
@mcp.tool()
def g61_create_user_data_pack(id: str) -> dict[str, Any] | list[Any]:
    """Subscribe user to a data pack (`POST /me/datapacks`, operationId: `createUserDataPack`)."""
    return Garage61Client().create_user_data_pack(id=id)


# 16. DELETE /me/datapacks/{id} (operationId: deleteUserDataPack)
@mcp.tool()
def g61_delete_user_data_pack(id: str) -> dict[str, Any] | list[Any]:
    """Unsubscribe user from a data pack (`DELETE /me/datapacks/{id}`, operationId: `deleteUserDataPack`)."""
    return Garage61Client().delete_user_data_pack(id=id)


# 17. GET /me/statistics (operationId: getStatistics)
@mcp.tool()
def g61_get_statistics(
    start: str | None = None,
    end: str | None = None,
    cars: list[int] | str | int | None = None,
    tracks: list[int] | str | int | None = None,
) -> dict[str, Any] | list[Any]:
    """Get personal driving statistics (`GET /me/statistics`, operationId: `getStatistics`)."""
    return Garage61Client().get_statistics(
        start=start,
        end=end,
        cars=cars,
        tracks=tracks,
    )


# 18. GET /platforms (operationId: findPlatforms)
@mcp.tool()
def g61_find_platforms() -> dict[str, Any] | list[Any]:
    """Available platforms (`GET /platforms`, operationId: `findPlatforms`)."""
    return Garage61Client().find_platforms()


# 19. GET /teams (operationId: findTeams)
@mcp.tool()
def g61_find_teams() -> dict[str, Any] | list[Any]:
    """Joined teams (`GET /teams`, operationId: `findTeams`)."""
    return Garage61Client().find_teams()


# 20. GET /teams/{id} (operationId: getTeam)
@mcp.tool()
def g61_get_team(id: str) -> dict[str, Any] | list[Any]:
    """Get information about a specific team (`GET /teams/{id}`, operationId: `getTeam`)."""
    return Garage61Client().get_team(id=id)


# 21. GET /teams/{team}/datapackgroups (operationId: getTeamDataPackGroups)
@mcp.tool()
def g61_get_team_data_pack_groups(
    team: str,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any] | list[Any]:
    """Get data pack groups for a team (`GET /teams/{team}/datapackgroups`, operationId: `getTeamDataPackGroups`)."""
    return Garage61Client().get_team_data_pack_groups(
        team=team,
        limit=limit,
        offset=offset,
    )


# 22. GET /teams/{team}/datapackgroups/{id} (operationId: getTeamDataPackGroup)
@mcp.tool()
def g61_get_team_data_pack_group(team: str, id: str) -> dict[str, Any] | list[Any]:
    """Get data pack group for a team (`GET /teams/{team}/datapackgroups/{id}`, operationId: `getTeamDataPackGroup`)."""
    return Garage61Client().get_team_data_pack_group(team=team, id=id)


# 23. POST /teams/{team}/datapackgroups/{id}/members (operationId: createDataPackGroupMember)
@mcp.tool()
def g61_create_data_pack_group_member(
    team: str,
    id: str,
    data_pack_id: str,
) -> dict[str, Any] | list[Any]:
    """Add data pack to a data pack group (`POST /teams/{team}/datapackgroups/{id}/members`, operationId: `createDataPackGroupMember`)."""
    return Garage61Client().create_data_pack_group_member(
        team=team,
        id=id,
        data_pack_id=data_pack_id,
    )


# 24. DELETE /teams/{team}/datapackgroups/{id}/members/{member} (operationId: deleteDataPackGroupMember)
@mcp.tool()
def g61_delete_data_pack_group_member(
    team: str,
    id: str,
    member: str,
) -> dict[str, Any] | list[Any]:
    """Remove data pack from a data pack group (`DELETE /teams/{team}/datapackgroups/{id}/members/{member}`, operationId: `deleteDataPackGroupMember`)."""
    return Garage61Client().delete_data_pack_group_member(
        team=team,
        id=id,
        member=member,
    )


# 25. DELETE /teams/{team}/datapackgroups/{id}/subscribers/{subscriber} (operationId: deleteDataPackGroupSubscriber)
@mcp.tool()
def g61_delete_data_pack_group_subscriber(
    team: str,
    id: str,
    subscriber: str,
) -> dict[str, Any] | list[Any]:
    """Remove subscriber from a data pack group (`DELETE /teams/{team}/datapackgroups/{id}/subscribers/{subscriber}`, operationId: `deleteDataPackGroupSubscriber`)."""
    return Garage61Client().delete_data_pack_group_subscriber(
        team=team,
        id=id,
        subscriber=subscriber,
    )


# 26. GET /teams/{team}/datapacks (operationId: getTeamDataPacks)
@mcp.tool()
def g61_get_team_data_packs(
    team: str,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any] | list[Any]:
    """Get data packs for a team (`GET /teams/{team}/datapacks`, operationId: `getTeamDataPacks`)."""
    return Garage61Client().get_team_data_packs(
        team=team,
        limit=limit,
        offset=offset,
    )


# 27. GET /teams/{team}/datapacks/{id} (operationId: getTeamDataPack)
@mcp.tool()
def g61_get_team_data_pack(team: str, id: str) -> dict[str, Any] | list[Any]:
    """Get data pack for a team (`GET /teams/{team}/datapacks/{id}`, operationId: `getTeamDataPack`)."""
    return Garage61Client().get_team_data_pack(team=team, id=id)


# 28. POST /teams/{team}/datapacks/{id} (operationId: updateTeamDataPack)
@mcp.tool()
def g61_update_team_data_pack(
    team: str,
    id: str,
    name: str | None = None,
    archived: bool | None = None,
) -> dict[str, Any] | list[Any]:
    """Update data pack for a team (`POST /teams/{team}/datapacks/{id}`, operationId: `updateTeamDataPack`)."""
    return Garage61Client().update_team_data_pack(
        team=team,
        id=id,
        name=name,
        archived=archived,
    )


# 29. POST /teams/{team}/datapacks/{id}/content (operationId: createTeamDataPackContent)
@mcp.tool()
def g61_create_team_data_pack_content(
    team: str,
    id: str,
    video_url: str,
    content_type: str = "video",
    featured: bool = True,
) -> dict[str, Any] | list[Any]:
    """Add content to a data pack for a team (`POST /teams/{team}/datapacks/{id}/content`, operationId: `createTeamDataPackContent`)."""
    return Garage61Client().create_team_data_pack_content(
        team=team,
        id=id,
        video_url=video_url,
        content_type=content_type,
        featured=featured,
    )


# 30. POST /teams/{team}/datapacks/{id}/content/{item} (operationId: updateTeamDataPackContent)
@mcp.tool()
def g61_update_team_data_pack_content(
    team: str,
    id: str,
    item: str,
    video_url: str,
    content_type: str = "video",
) -> dict[str, Any] | list[Any]:
    """Update data pack content for a team (`POST /teams/{team}/datapacks/{id}/content/{item}`, operationId: `updateTeamDataPackContent`)."""
    return Garage61Client().update_team_data_pack_content(
        team=team,
        id=id,
        item=item,
        video_url=video_url,
        content_type=content_type,
    )


# 31. DELETE /teams/{team}/datapacks/{id}/content/{item} (operationId: deleteTeamDataPackContent)
@mcp.tool()
def g61_delete_team_data_pack_content(
    team: str,
    id: str,
    item: str,
) -> dict[str, Any] | list[Any]:
    """Delete data pack content for a team (`DELETE /teams/{team}/datapacks/{id}/content/{item}`, operationId: `deleteTeamDataPackContent`)."""
    return Garage61Client().delete_team_data_pack_content(
        team=team,
        id=id,
        item=item,
    )


# 32. GET /teams/{team}/datapacks/{id}/content/{item}/ghost.bin (operationId: getTeamDataPackContentGhostLapBin)
@mcp.tool()
def g61_get_team_data_pack_content_ghost_lap_bin(
    team: str,
    id: str,
    item: str,
) -> dict[str, Any]:
    """Download ghost lap file for a content item (`GET /teams/{team}/datapacks/{id}/content/{item}/ghost.bin`, operationId: `getTeamDataPackContentGhostLapBin`)."""
    raw = Garage61Client().get_team_data_pack_content_ghost_lap_bin(
        team=team,
        id=id,
        item=item,
    )
    return {
        "team": team,
        "data_pack_id": id,
        "item_id": item,
        "content_type": "application/octet-stream",
        "size_bytes": len(raw),
        "data_base64": base64.b64encode(raw).decode("ascii"),
    }


# 33. GET /teams/{team}/datapacks/{id}/content/{item}/lap.csv (operationId: getTeamDataPackContentLapCsv)
@mcp.tool()
def g61_get_team_data_pack_content_lap_csv(
    team: str,
    id: str,
    item: str,
) -> str:
    """Export telemetry for a lap content item (`GET /teams/{team}/datapacks/{id}/content/{item}/lap.csv`, operationId: `getTeamDataPackContentLapCsv`)."""
    return Garage61Client().get_team_data_pack_content_lap_csv(
        team=team,
        id=id,
        item=item,
    )


# 34. POST /teams/{team}/datapacks/{id}/content/{item}/move (operationId: moveTeamDataPackContent)
@mcp.tool()
def g61_move_team_data_pack_content(
    team: str,
    id: str,
    item: str,
    position: int,
) -> dict[str, Any] | list[Any]:
    """Move data pack content for a team (`POST /teams/{team}/datapacks/{id}/content/{item}/move`, operationId: `moveTeamDataPackContent`)."""
    return Garage61Client().move_team_data_pack_content(
        team=team,
        id=id,
        item=item,
        position=position,
    )


# 35. GET /teams/{team}/datapacks/{id}/content/{item}/replay.bin (operationId: getTeamDataPackContentReplayBin)
@mcp.tool()
def g61_get_team_data_pack_content_replay_bin(
    team: str,
    id: str,
    item: str,
) -> dict[str, Any]:
    """Download file for a replay content item (`GET /teams/{team}/datapacks/{id}/content/{item}/replay.bin`, operationId: `getTeamDataPackContentReplayBin`)."""
    raw = Garage61Client().get_team_data_pack_content_replay_bin(
        team=team,
        id=id,
        item=item,
    )
    return {
        "team": team,
        "data_pack_id": id,
        "item_id": item,
        "content_type": "application/octet-stream",
        "size_bytes": len(raw),
        "data_base64": base64.b64encode(raw).decode("ascii"),
    }


# 36. GET /teams/{team}/datapacks/{id}/content/{item}/setup.sto (operationId: getTeamDataPackContentSetupSto)
@mcp.tool()
def g61_get_team_data_pack_content_setup_sto(
    team: str,
    id: str,
    item: str,
) -> dict[str, Any]:
    """Download iRacing setup for a setup content item (`GET /teams/{team}/datapacks/{id}/content/{item}/setup.sto`, operationId: `getTeamDataPackContentSetupSto`)."""
    raw = Garage61Client().get_team_data_pack_content_setup_sto(
        team=team,
        id=id,
        item=item,
    )
    return {
        "team": team,
        "data_pack_id": id,
        "item_id": item,
        "content_type": "application/octet-stream",
        "size_bytes": len(raw),
        "data_base64": base64.b64encode(raw).decode("ascii"),
    }


# 37. DELETE /teams/{team}/datapacks/{id}/subscribers/{subscriber} (operationId: deleteDataPackSubscriber)
@mcp.tool()
def g61_delete_data_pack_subscriber(
    team: str,
    id: str,
    subscriber: str,
) -> dict[str, Any] | list[Any]:
    """Remove subscriber from a data pack (`DELETE /teams/{team}/datapacks/{id}/subscribers/{subscriber}`, operationId: `deleteDataPackSubscriber`)."""
    return Garage61Client().delete_data_pack_subscriber(
        team=team,
        id=id,
        subscriber=subscriber,
    )


# 38. POST /teams/{team}/invites (operationId: createTeamInvite)
@mcp.tool()
def g61_create_team_invite(
    team: str,
    name: str,
    max_usages: int = 0,
    expires_after: int = 0,
) -> dict[str, Any] | list[Any]:
    """Create team invite (`POST /teams/{team}/invites`, operationId: `createTeamInvite`)."""
    return Garage61Client().create_team_invite(
        team=team,
        name=name,
        max_usages=max_usages,
        expires_after=expires_after,
    )


# 39. DELETE /teams/{team}/members/{id} (operationId: deleteTeamMember)
@mcp.tool()
def g61_delete_team_member(team: str, id: str) -> dict[str, Any] | list[Any]:
    """Remove member from team (`DELETE /teams/{team}/members/{id}`, operationId: `deleteTeamMember`)."""
    return Garage61Client().delete_team_member(team=team, id=id)


# 40. GET /teams/{team}/statistics (operationId: getTeamStatistics)
@mcp.tool()
def g61_get_team_statistics(
    team: str,
    start: str | None = None,
    end: str | None = None,
    cars: list[int] | str | int | None = None,
    tracks: list[int] | str | int | None = None,
) -> dict[str, Any] | list[Any]:
    """Get team driving statistics (`GET /teams/{team}/statistics`, operationId: `getTeamStatistics`)."""
    return Garage61Client().get_team_statistics(
        team=team,
        start=start,
        end=end,
        cars=cars,
        tracks=tracks,
    )


# 41. GET /teams/{team}/trainingplans (operationId: getTeamTrainingPlans)
@mcp.tool()
def g61_get_team_training_plans(
    team: str,
    limit: int | None = None,
    offset: int | None = None,
) -> dict[str, Any] | list[Any]:
    """Get training plans for a team (`GET /teams/{team}/trainingplans`, operationId: `getTeamTrainingPlans`)."""
    return Garage61Client().get_team_training_plans(
        team=team,
        limit=limit,
        offset=offset,
    )


# 42. GET /teams/{team}/trainingplans/{id} (operationId: getTeamTrainingPlan)
@mcp.tool()
def g61_get_team_training_plan(team: str, id: str) -> dict[str, Any] | list[Any]:
    """Get training plan for a team (`GET /teams/{team}/trainingplans/{id}`, operationId: `getTeamTrainingPlan`)."""
    return Garage61Client().get_team_training_plan(team=team, id=id)


# 43. GET /tracks (operationId: findTracks)
@mcp.tool()
def g61_find_tracks() -> dict[str, Any] | list[Any]:
    """Available tracks (`GET /tracks`, operationId: `findTracks`)."""
    return Garage61Client().find_tracks()


def main() -> None:
    """Entrypoint for running the FastMCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()

