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
    exit_steer = float(slowest_corner.get("user_exit_steering_at_50pct_throttle_deg", 0.0))

    if scrub > 4.0:
        return ("mid", "understeer")
    if exit_steer > 15.0:
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


def main() -> None:
    """Entrypoint for running the FastMCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
