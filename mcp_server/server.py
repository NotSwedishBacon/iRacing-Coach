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
# Garage 61 API v1 MCP Tools (16 Core Driving Data, Content, User & Team Endpoints)
# =============================================================================


# 1. GET /car-groups (operationId: findCarGroups)
@mcp.tool()
def g61_find_car_groups() -> dict[str, Any] | list[Any]:
    """Available car groups (`GET /car-groups`, operationId: `findCarGroups`)."""
    return Garage61Client().find_car_groups()


# 2. GET /cars (operationId: findCars)
@mcp.tool()
def g61_find_cars() -> dict[str, Any] | list[Any]:
    """Available cars (`GET /cars`, operationId: `findCars`)."""
    return Garage61Client().find_cars()


# 3. GET /events (operationId: findEvents)
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


# 4. GET /events/{id} (operationId: getEvent)
@mcp.tool()
def g61_get_event(id: str) -> dict[str, Any] | list[Any]:
    """Get event details (`GET /events/{id}`, operationId: `getEvent`)."""
    return Garage61Client().get_event(id=id)


# 5. GET /laps (operationId: findLaps)
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


# 6. GET /laps/{id} (operationId: getLap)
@mcp.tool()
def g61_get_lap(id: str) -> dict[str, Any] | list[Any]:
    """Get information about a specific lap (`GET /laps/{id}`, operationId: `getLap`)."""
    return Garage61Client().get_lap(id=id)


# 7. GET /laps/{id}/csv (operationId: getLapCSV)
@mcp.tool()
def g61_get_lap_csv(id: str) -> str:
    """Export the telemetry for a lap as a CSV file (`GET /laps/{id}/csv`, operationId: `getLapCSV`)."""
    return Garage61Client().get_lap_csv(id=id)


# 8. GET /laps/{id}/ghost.bin (operationId: getLapGhost)
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


# 9. GET /me (operationId: me)
@mcp.tool()
def g61_get_me() -> dict[str, Any] | list[Any]:
    """Information about the currently authenticated user (`GET /me`, operationId: `me`)."""
    return Garage61Client().get_me()


# 10. GET /me/accounts (operationId: getAccounts)
@mcp.tool()
def g61_get_accounts(rating_history: bool | None = None) -> dict[str, Any] | list[Any]:
    """Get linked accounts for the current user (`GET /me/accounts`, operationId: `getAccounts`)."""
    return Garage61Client().get_accounts(rating_history=rating_history)


# 11. GET /me/statistics (operationId: getStatistics)
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


# 12. GET /platforms (operationId: findPlatforms)
@mcp.tool()
def g61_find_platforms() -> dict[str, Any] | list[Any]:
    """Available platforms (`GET /platforms`, operationId: `findPlatforms`)."""
    return Garage61Client().find_platforms()


# 13. GET /teams (operationId: findTeams)
@mcp.tool()
def g61_find_teams() -> dict[str, Any] | list[Any]:
    """Joined teams (`GET /teams`, operationId: `findTeams`)."""
    return Garage61Client().find_teams()


# 14. GET /teams/{id} (operationId: getTeam)
@mcp.tool()
def g61_get_team(id: str) -> dict[str, Any] | list[Any]:
    """Get information about a specific team (`GET /teams/{id}`, operationId: `getTeam`)."""
    return Garage61Client().get_team(id=id)


# 15. GET /teams/{team}/statistics (operationId: getTeamStatistics)
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


# 16. GET /tracks (operationId: findTracks)
@mcp.tool()
def g61_find_tracks() -> dict[str, Any] | list[Any]:
    """Available tracks (`GET /tracks`, operationId: `findTracks`)."""
    return Garage61Client().find_tracks()


def main() -> None:
    """Entrypoint for running the FastMCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()

