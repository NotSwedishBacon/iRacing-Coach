"""Deterministic NumPy/SciPy Telemetry Analysis Engine (`telemetry/signal_math.py`).

Implements spatial distance alignment onto a common 1.0-meter distance axis
s in [0, TrackLength] via `scipy.interpolate.interp1d` and 15 deterministic physics
diagnostics per corner/sector to isolate driver technique errors vs. vehicle balance flaws:

Core 7 Physics Checks:
1. Braking Rate (`lazy_brake_hit`):
   Initial brake rise-time Delta t_rise to reach >= 80% pressure.
   If user_rise > ref_rise + 0.10s, flags "lazy brake hit".
2. Trail-Off Gradient (`abrupt_brake_release`):
   |d(Brake)/ds| between turn-in and apex.
   If user dumps the brake pedal (> 10%/m release rate) vs smooth bleed,
   flags "abrupt brake release unweighting front axle".
3. Coasting / Hesitation (`dead_time_coasting`):
   Contiguous meters where Brake < 0.05 and Throttle < 0.05 between turn-in and
   throttle pickup. If user coasting exceeds reference by > 8m, flags "dead-time coasting".
4. Friction Circle G-Sum (`g_sum_sag`):
   G_total = sqrt(a_x^2 + a_y^2). Checks for G-sum sags (< 1.0G) during pedal transition.
5. Steering Scrub Index (`steering_scrub`):
   Delta delta = |delta_user| - |delta_ref| from turn-in to apex.
   If Delta delta > 12 deg and v_min,user < v_min,ref, flags "steering scrub / thermal understeer".
6. Apex Delta (`early_apex_pinch`):
   Computes Delta v_min and spatial offset Delta s_apex.
   If Delta s_apex < -10m, flags "early apex pinch choking exit".
7. Exit Traction (`premature_throttle_lock`):
   Checks steering angle when throttle crosses 50%.
   If |delta| > 25 deg, flags "premature throttle application with excessive lock".

Extended 8 Advanced Technique & Vehicle Dynamics Metrics:
8. Phase Time Loss Decomposition (`time_loss_entry_s`, `time_loss_mid_s`, `time_loss_exit_s`, `time_loss_straight_s`):
   Integrates spatial time-loss density int (1/v_user - 1/v_ref) ds across entry, mid-corner,
   exit, and following straight carry, plus ranks top time-loss corners.
9. Braking Point Offset & Trail-Brake Overlap (`early_braking_overslow`, `late_braking_overshoot`, `missing_trail_brake_overlap`):
   Computes braking onset distance offset Delta s_brake and the simultaneous braking+cornering
   overlap integral I_trail = int_{turn-in}^{apex} Brake(s) * |a_y(s)| ds.
10. Steering Smoothness / Mid-Corner Sawing (`steering_sawing`):
    Computes RMS steering rate (deg/s) and direction reversals (d(delta)/ds sign changes)
    to isolate mid-corner micro-corrections.
11. Throttle Progression / Pedal Pumping (`throttle_pumping`):
    Computes post-apex negative throttle derivative integral sum(max(0, -d(Throttle)/ds))
    between apex and full throttle to catch lift/pump hesitation on exit.
12. Kinematic Curvature & Understeer/Oversteer Ratio (`snap_oversteer_countersteer`, `understeer_index_delta`):
    Computes path curvature kappa(s) = a_y / v^2 (or YawRate / v) and steering-to-curvature
    ratio K_us(s) = |delta(s)| / (L * |kappa(s)|), plus counter-steer detection (sign(delta) != sign(a_y)).
13. Friction Ellipse Grip Utilization % (`user_grip_utilization_pct`, `ref_grip_utilization_pct`):
    Normalizes combined longitudinal/lateral G against the 98th-percentile lap grip envelope
    (G_x,max, G_y,max): eta_grip = sqrt((a_x/G_x,max)^2 + (a_y/G_y,max)^2) * 100%.
14. V-Style vs. U-Style Exit Speed Trade-Off (`overslow_exit_compromise`):
    Compares apex minimum speed delta (Delta v_min) against track-out speed delta (Delta v_exit)
    and full-throttle spatial offset (Delta s_WOT) to catch rolling excess mid-corner speed that
    compromises exit acceleration.
15. Optional iRacing Channel Diagnostics (`excessive_abs_Intervention`, `late_mid_corner_downshift`):
    Measures ABS active distance (meters) during braking and detects destabilizing downshifts
    occurring after turn-in or wrong apex gear selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import interp1d

from telemetry.track_sectors import CornerSegment, get_track_corners

REQUIRED_CHANNELS = (
    "distance_m",
    "time_s",
    "speed",
    "brake",
    "throttle",
    "steering_deg",
    "lat_accel_g",
    "lon_accel_g",
)

OPTIONAL_CHANNELS = (
    "yaw_rate_rad_s",
    "abs_active",
    "gear",
    "rpm",
)


@dataclass
class DetectedFlaw:
    """Single deterministic driving or handling flaw detected in a corner."""

    code: str
    label: str
    phase: str
    symptom: str
    is_driver_error: bool
    metric_value: float
    reference_value: float
    threshold: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CornerAnalysisResult:
    """Deterministic mathematical telemetry deltas and flags for a single corner."""

    corner_number: int
    corner_name: str
    start_m: float
    end_m: float
    # Core 7 metrics
    user_brake_rise_time_s: float
    ref_brake_rise_time_s: float
    brake_rise_delta_s: float
    user_trail_off_gradient_pct_per_m: float
    ref_trail_off_gradient_pct_per_m: float
    user_coasting_distance_m: float
    ref_coasting_distance_m: float
    coasting_delta_m: float
    user_min_g_sum_transition: float
    ref_min_g_sum_transition: float
    steering_scrub_delta_deg: float
    user_v_min: float
    ref_v_min: float
    delta_v_min: float
    user_apex_s_m: float
    ref_apex_s_m: float
    delta_s_apex_m: float
    user_exit_steering_at_50pct_throttle_deg: float
    ref_exit_steering_at_50pct_throttle_deg: float
    # Core 7 boolean flags
    lazy_brake_hit: bool = False
    trail_off_warning: bool = False
    dead_time_coasting: bool = False
    g_sum_sag: bool = False
    steering_scrub: bool = False
    early_apex_pinch: bool = False
    premature_throttle_lock: bool = False
    # Extended 8 metrics
    time_loss_total_s: float = 0.0
    time_loss_entry_s: float = 0.0
    time_loss_mid_s: float = 0.0
    time_loss_exit_s: float = 0.0
    time_loss_straight_s: float = 0.0
    user_brake_onset_s_m: float = 0.0
    ref_brake_onset_s_m: float = 0.0
    brake_point_offset_m: float = 0.0
    user_trail_brake_overlap_gs_m: float = 0.0
    ref_trail_brake_overlap_gs_m: float = 0.0
    user_steering_rate_rms_deg_per_s: float = 0.0
    ref_steering_rate_rms_deg_per_s: float = 0.0
    user_steering_reversals: int = 0
    ref_steering_reversals: int = 0
    user_throttle_hesitation_pct: float = 0.0
    ref_throttle_hesitation_pct: float = 0.0
    understeer_index_delta: float = 0.0
    countersteer_distance_m: float = 0.0
    user_grip_utilization_pct: float = 0.0
    ref_grip_utilization_pct: float = 0.0
    grip_utilization_delta_pct: float = 0.0
    user_v_exit: float = 0.0
    ref_v_exit: float = 0.0
    delta_v_exit: float = 0.0
    full_throttle_offset_m: float = 0.0
    corner_line_style: str = "matched"
    user_abs_active_distance_m: float = 0.0
    ref_abs_active_distance_m: float = 0.0
    user_min_gear: int = 0
    ref_min_gear: int = 0
    user_late_downshifts_after_turn_in: int = 0
    # Extended boolean flags
    early_braking_overslow: bool = False
    late_braking_overshoot: bool = False
    missing_trail_brake_overlap: bool = False
    steering_sawing: bool = False
    throttle_pumping: bool = False
    snap_oversteer_countersteer: bool = False
    overslow_exit_compromise: bool = False
    excessive_abs_intervention: bool = False
    late_mid_corner_downshift: bool = False
    # Decision gate & flaws
    technique_fault_isolated: bool = False
    flaws: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LapAnalysisReport:
    """Aggregated telemetry math analysis report across corners."""

    track_length_m: float
    technique_fault_isolated: bool
    flaws: list[dict[str, Any]]
    corners: list[dict[str, Any]]
    summary_deltas: dict[str, Any]
    top_time_loss_corners: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_channel_dict(lap_data: Any) -> dict[str, np.ndarray]:
    """Convert a pandas DataFrame or dict of arrays into normalized 1D NumPy arrays."""
    if hasattr(lap_data, "to_dict") and hasattr(lap_data, "columns"):
        from telemetry.g61_client import normalize_telemetry_columns

        df = normalize_telemetry_columns(lap_data)
        raw: Mapping[str, Any] = {col: df[col].to_numpy(dtype=float) for col in df.columns if np.issubdtype(df[col].dtype, np.number)}
    elif isinstance(lap_data, Mapping):
        raw = {k: np.asarray(v, dtype=float) for k, v in lap_data.items()}
    else:
        raise TypeError(f"Unsupported lap_data type: {type(lap_data)!r}")

    if "distance_m" not in raw:
        raise ValueError("Telemetry data must include 'distance_m' (or 'LapDist').")

    s = np.asarray(raw["distance_m"], dtype=float)
    n = s.size
    if n < 2:
        raise ValueError("Telemetry data must contain at least 2 spatial samples.")

    speed = np.asarray(raw.get("speed", np.full(n, 40.0)), dtype=float)

    if "time_s" in raw:
        time_s = np.asarray(raw["time_s"], dtype=float)
    else:
        # Derive elapsed time from ds / v when time_s is omitted
        ds = np.diff(s, prepend=s[0])
        safe_v = np.maximum(speed, 1.0)
        time_s = np.cumsum(ds / safe_v)

    brake = np.asarray(raw.get("brake", np.zeros(n)), dtype=float)
    if np.nanmax(brake) > 1.5:
        brake = brake / 100.0

    throttle = np.asarray(raw.get("throttle", np.zeros(n)), dtype=float)
    if np.nanmax(throttle) > 1.5:
        throttle = throttle / 100.0

    steering_deg = np.asarray(raw.get("steering_deg", np.zeros(n)), dtype=float)
    lat_accel_g = np.asarray(raw.get("lat_accel_g", np.zeros(n)), dtype=float)
    lon_accel_g = np.asarray(raw.get("lon_accel_g", np.zeros(n)), dtype=float)

    out: dict[str, np.ndarray] = {
        "distance_m": s,
        "time_s": time_s,
        "speed": speed,
        "brake": brake,
        "throttle": throttle,
        "steering_deg": steering_deg,
        "lat_accel_g": lat_accel_g,
        "lon_accel_g": lon_accel_g,
    }
    for opt_col in OPTIONAL_CHANNELS:
        if opt_col in raw and np.asarray(raw[opt_col]).size == n:
            out[opt_col] = np.asarray(raw[opt_col], dtype=float)

    return out


def resample_lap_spatial(
    lap_data: Any,
    track_length_m: float | None = None,
    step_m: float = 1.0,
    s_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Interpolate a lap onto a uniform 1.0-meter spatial distance axis s in [0, TrackLength].

    Uses `scipy.interpolate.interp1d` with linear interpolation (nearest for discrete gear)
    and boundary clamping.
    """
    channels = _to_channel_dict(lap_data)
    s_raw = channels["distance_m"]

    # Ensure strictly increasing distance for scipy.interpolate.interp1d
    sort_idx = np.argsort(s_raw, kind="mergesort")
    s_sorted = s_raw[sort_idx]
    unique_mask = np.concatenate(([True], np.diff(s_sorted) > 1e-6))
    s_unique = s_sorted[unique_mask]

    if s_grid is None:
        min_s = float(np.ceil(s_unique[0])) if s_unique[0] > 0 else 0.0
        max_s = float(track_length_m) if track_length_m is not None else float(np.floor(s_unique[-1]))
        s_grid = np.arange(min_s, max_s + step_m * 0.5, step_m, dtype=float)

    resampled: dict[str, np.ndarray] = {"distance_m": s_grid}
    for name, values in channels.items():
        if name == "distance_m":
            continue
        v_unique = values[sort_idx][unique_mask]
        interp_kind = "nearest" if name == "gear" else "linear"
        interpolator = interp1d(
            s_unique,
            v_unique,
            kind=interp_kind,
            bounds_error=False,
            fill_value=(float(v_unique[0]), float(v_unique[-1])),
            assume_sorted=True,
        )
        resampled[name] = np.asarray(interpolator(s_grid), dtype=float)

    return resampled


def align_laps_spatially(
    user_lap: Any,
    ref_lap: Any,
    track_length_m: float | None = None,
    step_m: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Interpolate both user and reference laps onto an identical 1.0m distance axis."""
    u_ch = _to_channel_dict(user_lap)
    r_ch = _to_channel_dict(ref_lap)

    min_s = float(max(np.min(u_ch["distance_m"]), np.min(r_ch["distance_m"])))
    if min_s < 1.0:
        min_s = 0.0
    if track_length_m is None:
        max_s = float(min(np.max(u_ch["distance_m"]), np.max(r_ch["distance_m"])))
    else:
        max_s = float(track_length_m)

    s_grid = np.arange(np.floor(min_s), np.floor(max_s) + step_m * 0.5, step_m, dtype=float)
    return (
        resample_lap_spatial(u_ch, s_grid=s_grid, step_m=step_m),
        resample_lap_spatial(r_ch, s_grid=s_grid, step_m=step_m),
    )


def _compute_brake_rise_time(
    time_s: np.ndarray,
    brake: np.ndarray,
    onset_threshold: float = 0.05,
    target_pressure: float = 0.80,
) -> float:
    """Calculate initial brake rise-time Delta t_rise to reach >= 80% brake pressure.

    If peak brake pressure in the zone is below 80% (e.g. a medium braking corner),
    evaluates rise time to 80% of the driver's peak brake application in that zone.
    """
    if brake.size == 0 or float(np.max(brake)) < onset_threshold:
        return 0.0

    onset_indices = np.where(brake >= onset_threshold)[0]
    if onset_indices.size == 0:
        return 0.0
    start_idx = int(onset_indices[0])
    # Walk back one sample if available to capture the true start of pedal movement
    if start_idx > 0 and brake[start_idx - 1] < onset_threshold:
        t_start = float(
            np.interp(
                onset_threshold,
                [brake[start_idx - 1], brake[start_idx]],
                [time_s[start_idx - 1], time_s[start_idx]],
            )
        )
    else:
        t_start = float(time_s[start_idx])

    peak_val = float(np.max(brake[start_idx:]))
    effective_target = target_pressure if peak_val >= target_pressure else 0.80 * peak_val

    target_indices = np.where(brake[start_idx:] >= effective_target)[0]
    if target_indices.size == 0:
        peak_idx = start_idx + int(np.argmax(brake[start_idx:]))
        return max(0.0, float(time_s[peak_idx]) - t_start)

    hit_idx = start_idx + int(target_indices[0])
    if hit_idx > start_idx and brake[hit_idx - 1] < effective_target:
        t_hit = float(
            np.interp(
                effective_target,
                [brake[hit_idx - 1], brake[hit_idx]],
                [time_s[hit_idx - 1], time_s[hit_idx]],
            )
        )
    else:
        t_hit = float(time_s[hit_idx])

    return max(0.0, t_hit - t_start)


def _find_turn_in_index(
    steering_deg: np.ndarray,
    lat_accel_g: np.ndarray,
    brake: np.ndarray,
    apex_idx: int,
) -> int:
    """Locate the turn-in index preceding the apex (at or after peak brake pressure if braking)."""
    if apex_idx <= 1:
        return 0

    pre_apex_brake = brake[: apex_idx + 1]
    min_start = 0
    if np.max(pre_apex_brake) > 0.10:
        min_start = int(np.argmax(pre_apex_brake))

    search_steer = np.abs(steering_deg[min_start : apex_idx + 1])
    search_ay = np.abs(lat_accel_g[min_start : apex_idx + 1])

    candidates = np.where((search_steer >= 10.0) | (search_ay >= 0.35))[0]
    if candidates.size > 0:
        return min_start + int(candidates[0])

    return min_start


def _compute_trail_off_gradient(
    s: np.ndarray,
    brake: np.ndarray,
    turn_in_idx: int,
    apex_idx: int,
) -> float:
    """Calculate maximum brake release gradient |d(Brake)/ds| (in %/m) between turn-in and apex.

    Returns the peak release rate (where d(Brake)/ds < 0) in percentage points per meter
    (0..100 scale per meter) so that > 10%/m (0.10/m) indicates dumping the brake pedal.
    """
    start = min(turn_in_idx, apex_idx)
    end = max(turn_in_idx, apex_idx)
    if end - start < 1:
        start = 0
        end = max(1, apex_idx)

    # Also include region from peak brake to apex in case the driver dumps the brake just before turn-in
    if np.max(brake[: apex_idx + 1]) > 0.05:
        peak_brake_idx = int(np.argmax(brake[: apex_idx + 1]))
        start = min(start, peak_brake_idx)

    seg_s = s[start : end + 1]
    seg_brake = brake[start : end + 1]
    if seg_s.size < 2:
        return 0.0

    ds = np.diff(seg_s)
    d_brake = np.diff(seg_brake)
    valid = ds > 1e-6
    if not np.any(valid):
        return 0.0

    # Release gradients correspond to decreasing brake pressure (d_brake < 0)
    release_rates_per_m = np.where(d_brake[valid] < 0.0, -d_brake[valid] / ds[valid], 0.0)
    if release_rates_per_m.size == 0:
        return 0.0

    # Return in %/m (e.g. 0.12 per meter -> 12.0 %/m)
    return float(np.max(release_rates_per_m) * 100.0)


def _max_contiguous_coasting_meters(
    s: np.ndarray,
    brake: np.ndarray,
    throttle: np.ndarray,
    turn_in_idx: int,
    apex_idx: int,
) -> float:
    """Measure maximum contiguous meters where Brake < 0.05 and Throttle < 0.05 between turn-in and throttle pickup."""
    # Find throttle pickup at or after apex (where throttle >= 0.05 and stays up)
    post_turn_in_throttle = throttle[turn_in_idx:]
    pickup_candidates = np.where(post_turn_in_throttle >= 0.20)[0]
    if pickup_candidates.size > 0:
        end_idx = max(apex_idx, turn_in_idx + int(pickup_candidates[0]))
    else:
        end_idx = len(s) - 1

    start_idx = min(turn_in_idx, end_idx)
    seg_s = s[start_idx : end_idx + 1]
    seg_b = brake[start_idx : end_idx + 1]
    seg_t = throttle[start_idx : end_idx + 1]

    if seg_s.size < 2:
        return 0.0

    coasting_mask = (seg_b < 0.05) & (seg_t < 0.05)
    if not np.any(coasting_mask):
        return 0.0

    padded = np.concatenate(([False], coasting_mask, [False]))
    run_starts = np.where(padded[1:] & ~padded[:-1])[0]
    run_ends = np.where(~padded[1:] & padded[:-1])[0] - 1

    max_meters = 0.0
    for st, en in zip(run_starts, run_ends):
        # Measure contiguous distance in meters; each sample on 1m grid represents step_m
        meters = float(seg_s[en] - seg_s[st]) + float(np.median(np.diff(seg_s)))
        if meters > max_meters:
            max_meters = meters
    return max_meters


def _compute_min_g_sum_transition(
    lon_accel_g: np.ndarray,
    lat_accel_g: np.ndarray,
    turn_in_idx: int,
    apex_idx: int,
) -> float:
    """Compute G_total = sqrt(a_x^2 + a_y^2) and find the minimum during pedal transition (turn-in to apex)."""
    g_total = np.sqrt(np.square(lon_accel_g) + np.square(lat_accel_g))
    start = min(turn_in_idx, apex_idx)
    end = max(turn_in_idx, apex_idx)
    if end <= start:
        return float(np.min(g_total)) if g_total.size > 0 else 0.0
    return float(np.min(g_total[start : end + 1]))


def _compute_steering_scrub_delta(
    user_steer_deg: np.ndarray,
    ref_steer_deg: np.ndarray,
    turn_in_idx: int,
    apex_idx: int,
) -> float:
    """Compute Delta delta = |delta_user| - |delta_ref| from turn-in to apex."""
    start = min(turn_in_idx, apex_idx)
    end = max(turn_in_idx, apex_idx)
    if end < start:
        start, end = 0, len(user_steer_deg) - 1

    delta_arr = np.abs(user_steer_deg[start : end + 1]) - np.abs(ref_steer_deg[start : end + 1])
    if delta_arr.size == 0:
        return 0.0
    # Use the mean excess steering lock across the turn-in-to-apex window
    return float(np.mean(delta_arr))


def _compute_exit_steering_at_throttle_50(
    throttle: np.ndarray,
    steering_deg: np.ndarray,
    apex_idx: int,
) -> float:
    """Return |steering_deg| when throttle crosses 50% (0.50) at or after the apex."""
    post_apex_t = throttle[apex_idx:]
    post_apex_s = np.abs(steering_deg[apex_idx:])
    if post_apex_t.size == 0:
        return 0.0

    crossings = np.where(post_apex_t >= 0.50)[0]
    if crossings.size == 0:
        return 0.0
    idx = int(crossings[0])
    if idx > 0 and post_apex_t[idx - 1] < 0.50:
        return float(
            np.interp(
                0.50,
                [post_apex_t[idx - 1], post_apex_t[idx]],
                [post_apex_s[idx - 1], post_apex_s[idx]],
            )
        )
    return float(post_apex_s[idx])


# -----------------------------------------------------------------------------
# Extended 8 Deterministic Physics Helper Functions
# -----------------------------------------------------------------------------


def _integrate_spatial_time_loss(
    s: np.ndarray,
    u_v: np.ndarray,
    r_v: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    """Integrate spatial time-loss density int_{s1}^{s2} (1/v_user - 1/v_ref) ds in seconds.

    Positive value means the user lost time relative to the reference lap; negative means user gained time.
    """
    st = max(0, min(start_idx, end_idx))
    en = min(len(s) - 1, max(start_idx, end_idx))
    if en <= st:
        return 0.0
    seg_s = s[st : en + 1]
    u_safe = np.maximum(u_v[st : en + 1], 1.0)
    r_safe = np.maximum(r_v[st : en + 1], 1.0)
    slowness_delta = (1.0 / u_safe) - (1.0 / r_safe)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is not None:
        return float(trapz_fn(slowness_delta, seg_s))
    return float(np.sum(0.5 * (slowness_delta[:-1] + slowness_delta[1:]) * np.diff(seg_s)))


def _find_brake_onset_distance(
    s: np.ndarray,
    brake: np.ndarray,
    apex_idx: int,
    onset_threshold: float = 0.05,
) -> float:
    """Find spatial distance s (in meters) where brake application crosses `onset_threshold` prior to apex."""
    pre_apex_b = brake[: apex_idx + 1]
    indices = np.where(pre_apex_b >= onset_threshold)[0]
    if indices.size == 0:
        return float(s[0])
    idx = int(indices[0])
    if idx > 0 and pre_apex_b[idx - 1] < onset_threshold:
        return float(
            np.interp(
                onset_threshold,
                [pre_apex_b[idx - 1], pre_apex_b[idx]],
                [s[idx - 1], s[idx]],
            )
        )
    return float(s[idx])


def _compute_trail_brake_overlap_integral(
    s: np.ndarray,
    brake: np.ndarray,
    lat_accel_g: np.ndarray,
    turn_in_idx: int,
    apex_idx: int,
) -> float:
    """Compute simultaneous braking + cornering overlap integral I_trail = int Brake(s) * |a_y(s)| ds."""
    st = min(turn_in_idx, apex_idx)
    en = max(turn_in_idx, apex_idx)
    if en <= st:
        return 0.0
    seg_s = s[st : en + 1]
    seg_b = np.where(brake[st : en + 1] >= 0.05, brake[st : en + 1], 0.0)
    seg_ay = np.where(np.abs(lat_accel_g[st : en + 1]) >= 0.35, np.abs(lat_accel_g[st : en + 1]), 0.0)
    integrand = seg_b * seg_ay
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is not None:
        return float(trapz_fn(integrand, seg_s))
    return float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * np.diff(seg_s)))


def _compute_steering_smoothness(
    s: np.ndarray,
    time_s: np.ndarray,
    steering_deg: np.ndarray,
    turn_in_idx: int,
    apex_idx: int,
) -> tuple[float, int]:
    """Compute RMS steering wheel angular rate (deg/s) and direction reversals (sawing count).

    In a clean single-arc corner, d(steering)/ds has at most 1 major sign change (at peak lock).
    Additional sign changes above a deadband of 0.5 deg/m indicate mid-corner steering corrections.
    """
    end_window = min(len(s) - 1, apex_idx + max(5, (apex_idx - turn_in_idx) // 2))
    st = max(0, min(turn_in_idx, end_window))
    en = max(st + 1, end_window)
    seg_s = s[st : en + 1]
    seg_t = time_s[st : en + 1]
    seg_steer = steering_deg[st : en + 1]
    if seg_s.size < 3:
        return 0.0, 0

    dt = np.diff(seg_t)
    d_steer = np.diff(seg_steer)
    valid_dt = dt > 1e-5
    if np.any(valid_dt):
        rates_deg_s = d_steer[valid_dt] / dt[valid_dt]
        rms_rate = float(np.sqrt(np.mean(np.square(rates_deg_s))))
    else:
        rms_rate = 0.0

    ds = np.diff(seg_s)
    slope_per_m = np.where(ds > 1e-5, d_steer / ds, 0.0)
    # Filter out tiny noise (< 0.45 deg/m) before counting direction reversals
    active_slopes = slope_per_m[np.abs(slope_per_m) >= 0.45]
    if active_slopes.size < 2:
        reversals = 0
    else:
        signs = np.sign(active_slopes)
        reversals = int(np.sum(signs[1:] != signs[:-1]))
    return rms_rate, reversals


def _compute_throttle_hesitation_pct(
    throttle: np.ndarray,
    apex_idx: int,
) -> float:
    """Measure cumulative negative throttle drops (in percentage points) between apex and full throttle (98%)."""
    post_apex = throttle[apex_idx:]
    if post_apex.size < 2:
        return 0.0
    full_Throttle_indices = np.where(post_apex >= 0.98)[0]
    if full_Throttle_indices.size > 0:
        end_idx = int(full_Throttle_indices[0]) + 1
    else:
        end_idx = len(post_apex)
    seg_t = post_apex[: max(2, end_idx)]
    d_t = np.diff(seg_t)
    # Sum all negative drops where the driver lifted after starting throttle application
    drops = np.where(d_t < -0.01, -d_t, 0.0)
    return float(np.sum(drops) * 100.0)


def _compute_understeer_and_countersteer(
    s: np.ndarray,
    speed: np.ndarray,
    steering_deg: np.ndarray,
    lat_accel_g: np.ndarray,
    turn_in_idx: int,
    apex_idx: int,
    yaw_rate_rad_s: np.ndarray | None = None,
    wheelbase_m: float = 2.75,
) -> tuple[float, float]:
    """Compute neutral-steer normalized understeer index and counter-steer distance (m).

    - Path curvature kappa(s) = |YawRate(s)| / v(s) when YawRate is present, else |a_y(s)| * 9.80665 / v(s)^2.
    - Kinematic steering ratio: |delta_rad(s)| / (wheelbase_m * kappa(s)).
    - Counter-steer distance: meters in the corner where |a_y| >= 0.4G, |steering_deg| >= 5 deg,
      and sign(steering_deg) != sign(lat_accel_g).
    """
    st = min(turn_in_idx, apex_idx)
    en = max(turn_in_idx, apex_idx)
    if en <= st:
        st, en = 0, len(s) - 1

    v_safe = np.maximum(speed[st : en + 1], 5.0)
    ay_seg = lat_accel_g[st : en + 1]
    steer_seg = steering_deg[st : en + 1]

    if yaw_rate_rad_s is not None and yaw_rate_rad_s.size == len(s):
        kappa = np.abs(yaw_rate_rad_s[st : en + 1]) / v_safe
    else:
        kappa = (np.abs(ay_seg) * 9.80665) / np.square(v_safe)

    active = (np.abs(ay_seg) >= 0.40) & (kappa > 1e-4)
    if np.any(active):
        steer_rad = np.radians(np.abs(steer_seg[active]))
        us_ratio = float(np.median(steer_rad / (wheelbase_m * kappa[active])))
    else:
        us_ratio = 0.0

    # Counter-steer check across the full corner segment (including exit)
    cornering_mask = (np.abs(lat_accel_g) >= 0.40) & (np.abs(steering_deg) >= 5.0)
    opposite_sign = (np.sign(steering_deg) * np.sign(lat_accel_g)) < 0.0
    cs_mask = cornering_mask & opposite_sign
    step_m = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    countersteer_m = float(np.sum(cs_mask) * step_m)
    return us_ratio, countersteer_m


def _compute_grip_utilization_pct(
    lon_accel_g: np.ndarray,
    lat_accel_g: np.ndarray,
    gx_max: float,
    gy_max: float,
    turn_in_idx: int,
    apex_idx: int,
) -> float:
    """Compute normalized friction ellipse grip utilization % across the cornering window."""
    end_idx = min(len(lon_accel_g) - 1, apex_idx + max(5, (apex_idx - turn_in_idx) // 2))
    st = max(0, min(turn_in_idx, end_idx))
    en = max(st + 1, end_idx)
    ax = lon_accel_g[st : en + 1]
    ay = lat_accel_g[st : en + 1]
    norm_g = np.sqrt(np.square(ax / max(gx_max, 0.5)) + np.square(ay / max(gy_max, 0.5)))
    return float(np.mean(norm_g) * 100.0)


def _find_full_throttle_distance(
    s: np.ndarray,
    throttle: np.ndarray,
    apex_idx: int,
    wot_threshold: float = 0.95,
) -> float:
    """Find distance s (in meters) at or after apex where throttle reaches >= 95%."""
    post_apex = throttle[apex_idx:]
    indices = np.where(post_apex >= wot_threshold)[0]
    if indices.size == 0:
        return float(s[-1])
    return float(s[apex_idx + int(indices[0])])


def _compute_abs_and_downshift_metrics(
    s: np.ndarray,
    brake: np.ndarray,
    turn_in_idx: int,
    apex_idx: int,
    abs_active: np.ndarray | None = None,
    gear: np.ndarray | None = None,
) -> tuple[float, int, int]:
    """Compute ABS active distance (m), minimum apex gear, and late downshifts after turn-in."""
    step_m = float(np.median(np.diff(s))) if len(s) > 1 else 1.0
    if abs_active is not None and abs_active.size == len(s):
        abs_mask = (brake >= 0.10) & (abs_active >= 0.5)
        abs_m = float(np.sum(abs_mask) * step_m)
    else:
        abs_m = 0.0

    if gear is not None and gear.size == len(s):
        rounded_gear = np.rint(gear).astype(int)
        min_gear = int(np.min(rounded_gear[: apex_idx + 1])) if apex_idx >= 0 else int(np.min(rounded_gear))
        # Count downshifts occurring between turn_in_idx + 3m and apex_idx
        late_start = min(turn_in_idx + 3, apex_idx)
        if apex_idx > late_start:
            gear_changes = np.diff(rounded_gear[late_start : apex_idx + 1])
            late_downshifts = int(np.sum(gear_changes < 0))
        else:
            late_downshifts = 0
    else:
        min_gear = 0
        late_downshifts = 0

    return abs_m, min_gear, late_downshifts


def analyze_corner_segment(
    user_aligned: Mapping[str, np.ndarray],
    ref_aligned: Mapping[str, np.ndarray],
    segment: CornerSegment,
    next_segment_start_m: float | None = None,
    gx_max: float = 1.50,
    gy_max: float = 1.60,
) -> CornerAnalysisResult:
    """Run all 15 deterministic physics checks on a single corner segment."""
    s_full = user_aligned["distance_m"]
    mask = (s_full >= segment.start_m) & (s_full <= segment.end_m)
    if np.sum(mask) < 3:
        mask = np.ones_like(s_full, dtype=bool)

    s = s_full[mask]
    u_t = user_aligned["time_s"][mask]
    r_t = ref_aligned["time_s"][mask]
    u_v = user_aligned["speed"][mask]
    r_v = ref_aligned["speed"][mask]
    u_b = user_aligned["brake"][mask]
    r_b = ref_aligned["brake"][mask]
    u_th = user_aligned["throttle"][mask]
    r_th = ref_aligned["throttle"][mask]
    u_st = user_aligned["steering_deg"][mask]
    r_st = ref_aligned["steering_deg"][mask]
    u_ay = user_aligned["lat_accel_g"][mask]
    r_ay = ref_aligned["lat_accel_g"][mask]
    u_ax = user_aligned["lon_accel_g"][mask]
    r_ax = ref_aligned["lon_accel_g"][mask]

    u_yaw = user_aligned["yaw_rate_rad_s"][mask] if "yaw_rate_rad_s" in user_aligned else None
    r_yaw = ref_aligned["yaw_rate_rad_s"][mask] if "yaw_rate_rad_s" in ref_aligned else None
    u_abs = user_aligned["abs_active"][mask] if "abs_active" in user_aligned else None
    r_abs = ref_aligned["abs_active"][mask] if "abs_active" in ref_aligned else None
    u_gear = user_aligned["gear"][mask] if "gear" in user_aligned else None
    r_gear = ref_aligned["gear"][mask] if "gear" in ref_aligned else None

    # 6. Apex Delta: local minimum speed v_min and spatial offset s_apex
    u_apex_idx = int(np.argmin(u_v))
    r_apex_idx = int(np.argmin(r_v))
    u_v_min = float(u_v[u_apex_idx])
    r_v_min = float(r_v[r_apex_idx])
    delta_v_min = u_v_min - r_v_min

    u_apex_s = float(s[u_apex_idx])
    r_apex_s = float(s[r_apex_idx])
    delta_s_apex = u_apex_s - r_apex_s

    # Turn-in index
    u_turn_in_idx = _find_turn_in_index(u_st, u_ay, u_b, u_apex_idx)
    r_turn_in_idx = _find_turn_in_index(r_st, r_ay, r_b, r_apex_idx)
    common_turn_in = min(u_turn_in_idx, r_turn_in_idx)
    common_apex = max(u_apex_idx, r_apex_idx)

    # 1. Braking Rate: initial brake rise-time Delta t_rise to reach >= 80% pressure
    u_rise = _compute_brake_rise_time(u_t, u_b)
    r_rise = _compute_brake_rise_time(r_t, r_b)
    rise_delta = u_rise - r_rise

    # 2. Trail-Off Gradient: |d(Brake)/ds| between turn-in and apex (%/m)
    u_trail_grad = _compute_trail_off_gradient(s, u_b, u_turn_in_idx, u_apex_idx)
    r_trail_grad = _compute_trail_off_gradient(s, r_b, r_turn_in_idx, r_apex_idx)

    # 3. Coasting / Hesitation: contiguous meters where Brake < 0.05 and Throttle < 0.05
    u_coast_m = _max_contiguous_coasting_meters(s, u_b, u_th, u_turn_in_idx, u_apex_idx)
    r_coast_m = _max_contiguous_coasting_meters(s, r_b, r_th, r_turn_in_idx, r_apex_idx)
    coast_delta_m = u_coast_m - r_coast_m

    # 4. Friction Circle (G-Sum): G_total = sqrt(a_x^2 + a_y^2) during pedal transition
    u_min_g = _compute_min_g_sum_transition(u_ax, u_ay, u_turn_in_idx, u_apex_idx)
    r_min_g = _compute_min_g_sum_transition(r_ax, r_ay, r_turn_in_idx, r_apex_idx)

    # 5. Steering Scrub Index: |delta_user| - |delta_ref| from turn-in to apex
    scrub_delta_deg = _compute_steering_scrub_delta(u_st, r_st, common_turn_in, common_apex)

    # 7. Exit Traction: steering angle when throttle crosses 50%
    u_exit_steer = _compute_exit_steering_at_throttle_50(u_th, u_st, u_apex_idx)
    r_exit_steer = _compute_exit_steering_at_throttle_50(r_th, r_st, r_apex_idx)

    # 8. Phase Time Loss Decomposition (Entry, Mid-Corner, Exit, Following Straight)
    time_loss_entry_s = _integrate_spatial_time_loss(s, u_v, r_v, 0, common_turn_in)
    time_loss_mid_s = _integrate_spatial_time_loss(s, u_v, r_v, common_turn_in, r_apex_idx)
    time_loss_exit_s = _integrate_spatial_time_loss(s, u_v, r_v, r_apex_idx, len(s) - 1)
    time_loss_total_s = time_loss_entry_s + time_loss_mid_s + time_loss_exit_s

    time_loss_straight_s = 0.0
    if next_segment_start_m is not None and next_segment_start_m > segment.end_m:
        straight_mask = (s_full >= segment.end_m) & (s_full <= next_segment_start_m)
        if np.sum(straight_mask) >= 2:
            s_str = s_full[straight_mask]
            time_loss_straight_s = _integrate_spatial_time_loss(
                s_str,
                user_aligned["speed"][straight_mask],
                ref_aligned["speed"][straight_mask],
                0,
                len(s_str) - 1,
            )

    # 9. Braking Point Offset & Trail-Brake Overlap Integral
    u_brake_onset_s = _find_brake_onset_distance(s, u_b, u_apex_idx)
    r_brake_onset_s = _find_brake_onset_distance(s, r_b, r_apex_idx)
    has_braking_zone = bool(np.max(r_b[: r_apex_idx + 1]) >= 0.15 and np.max(u_b[: u_apex_idx + 1]) >= 0.15)
    brake_point_offset_m = (u_brake_onset_s - r_brake_onset_s) if has_braking_zone else 0.0

    u_trail_overlap = _compute_trail_brake_overlap_integral(s, u_b, u_ay, u_turn_in_idx, u_apex_idx)
    r_trail_overlap = _compute_trail_brake_overlap_integral(s, r_b, r_ay, r_turn_in_idx, r_apex_idx)

    # 10. Steering Smoothness / Mid-Corner Sawing
    u_steer_rms, u_reversals = _compute_steering_smoothness(s, u_t, u_st, u_turn_in_idx, u_apex_idx)
    r_steer_rms, r_reversals = _compute_steering_smoothness(s, r_t, r_st, r_turn_in_idx, r_apex_idx)

    # 11. Throttle Progression / Pedal Pumping
    u_th_hesitation = _compute_throttle_hesitation_pct(u_th, u_apex_idx)
    r_th_hesitation = _compute_throttle_hesitation_pct(r_th, r_apex_idx)

    # 12. Kinematic Curvature Understeer Index & Counter-Steer Detection
    u_us_idx, countersteer_m = _compute_understeer_and_countersteer(
        s, u_v, u_st, u_ay, u_turn_in_idx, u_apex_idx, yaw_rate_rad_s=u_yaw
    )
    r_us_idx, _ = _compute_understeer_and_countersteer(
        s, r_v, r_st, r_ay, r_turn_in_idx, r_apex_idx, yaw_rate_rad_s=r_yaw
    )
    understeer_index_delta = u_us_idx - r_us_idx

    # 13. Friction Ellipse Grip Utilization %
    u_grip_pct = _compute_grip_utilization_pct(u_ax, u_ay, gx_max, gy_max, u_turn_in_idx, u_apex_idx)
    r_grip_pct = _compute_grip_utilization_pct(r_ax, r_ay, gx_max, gy_max, r_turn_in_idx, r_apex_idx)
    grip_delta_pct = u_grip_pct - r_grip_pct

    # 14. V-Style vs U-Style Corner Exit Speed Trade-Off
    u_v_exit = float(u_v[-1])
    r_v_exit = float(r_v[-1])
    delta_v_exit = u_v_exit - r_v_exit
    u_wot_s = _find_full_throttle_distance(s, u_th, u_apex_idx)
    r_wot_s = _find_full_throttle_distance(s, r_th, r_apex_idx)
    wot_offset_m = u_wot_s - r_wot_s

    if delta_v_min > 1.0 and delta_v_exit < -1.0:
        corner_line_style = "overcharged_entry_compromised_exit"
    elif delta_v_min < -1.0 and delta_v_exit > 0.5:
        corner_line_style = "v_style_prioritized_exit"
    elif abs(delta_v_min) <= 1.0 and abs(delta_v_exit) <= 1.0:
        corner_line_style = "matched"
    else:
        corner_line_style = "deficit"

    # 15. Optional iRacing Channel Diagnostics (ABS & Mid-Corner Downshifts)
    u_abs_m, u_min_gear, u_late_shifts = _compute_abs_and_downshift_metrics(
        s, u_b, u_turn_in_idx, u_apex_idx, abs_active=u_abs, gear=u_gear
    )
    r_abs_m, r_min_gear, _ = _compute_abs_and_downshift_metrics(
        s, r_b, r_turn_in_idx, r_apex_idx, abs_active=r_abs, gear=r_gear
    )

    flaws: list[DetectedFlaw] = []

    # Check 1: Braking Rate
    is_lazy_brake = bool(rise_delta > 0.10)
    if is_lazy_brake:
        flaws.append(
            DetectedFlaw(
                code="lazy_brake_hit",
                label="lazy brake hit",
                phase="entry",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(u_rise, 4),
                reference_value=round(r_rise, 4),
                threshold=round(r_rise + 0.10, 4),
                explanation=(
                    f"Initial brake rise time to >=80% pressure took {u_rise:.2f}s vs reference "
                    f"{r_rise:.2f}s (+{rise_delta:.2f}s > +0.10s threshold), delaying pitch transfer."
                ),
            )
        )

    # Check 2: Trail-Off Gradient (> 10%/m release rate vs smooth bleed)
    is_trail_off_dump = bool(u_trail_grad > 10.0 and u_trail_grad > r_trail_grad * 1.25)
    if is_trail_off_dump:
        flaws.append(
            DetectedFlaw(
                code="abrupt_brake_release",
                label="abrupt brake release unweighting front axle",
                phase="entry",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(u_trail_grad, 3),
                reference_value=round(r_trail_grad, 3),
                threshold=10.0,
                explanation=(
                    f"Brake trail-off gradient reached {u_trail_grad:.1f}%/m vs reference "
                    f"{r_trail_grad:.1f}%/m (>10%/m threshold), abruptly unweighting the front tire contact patch."
                ),
            )
        )

    # Check 3: Coasting / Hesitation (> 8m over reference)
    is_coasting = bool(coast_delta_m > 8.0)
    if is_coasting:
        flaws.append(
            DetectedFlaw(
                code="dead_time_coasting",
                label="dead-time coasting",
                phase="mid",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(u_coast_m, 2),
                reference_value=round(r_coast_m, 2),
                threshold=round(r_coast_m + 8.0, 2),
                explanation=(
                    f"Contiguous pedal dead-time (Brake < 5% & Throttle < 5%) spanned {u_coast_m:.1f}m "
                    f"vs reference {r_coast_m:.1f}m (+{coast_delta_m:.1f}m > 8.0m threshold)."
                ),
            )
        )

    # Check 4: Friction Circle (G-Sum sag < 1.0G during pedal transition)
    is_g_sag = bool(u_min_g < 1.0 and (r_min_g >= 1.0 or u_min_g < r_min_g - 0.15))
    if is_g_sag:
        flaws.append(
            DetectedFlaw(
                code="g_sum_sag",
                label="G-sum sag during pedal transition",
                phase="entry",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(u_min_g, 3),
                reference_value=round(r_min_g, 3),
                threshold=1.0,
                explanation=(
                    f"Combined G-sum dropped to {u_min_g:.2f}G (< 1.0G threshold, reference {r_min_g:.2f}G) "
                    "during brake-to-turn-in transition, under-utilizing the tire friction circle."
                ),
            )
        )

    # Check 5: Steering Scrub Index (Delta delta > 12 deg and v_min,user < v_min,ref)
    is_steering_scrub = bool(scrub_delta_deg > 12.0 and u_v_min < r_v_min)
    if is_steering_scrub:
        flaws.append(
            DetectedFlaw(
                code="steering_scrub",
                label="steering scrub / thermal understeer",
                phase="mid",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(scrub_delta_deg, 2),
                reference_value=0.0,
                threshold=12.0,
                explanation=(
                    f"Steering input exceeded reference by +{scrub_delta_deg:.1f} deg (> 12.0 deg threshold) "
                    f"while apex speed was {abs(delta_v_min):.2f} m/s slower ({u_v_min:.1f} vs {r_v_min:.1f}), "
                    "scrubbing the front tires past peak slip angle."
                ),
            )
        )

    # Check 6: Apex Delta (Delta s_apex < -10m -> early apex pinch)
    is_early_apex = bool(delta_s_apex < -10.0)
    if is_early_apex:
        flaws.append(
            DetectedFlaw(
                code="early_apex_pinch",
                label="early apex pinch choking exit",
                phase="mid",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(delta_s_apex, 2),
                reference_value=0.0,
                threshold=-10.0,
                explanation=(
                    f"Minimum speed occurred at {u_apex_s:.1f}m vs reference {r_apex_s:.1f}m "
                    f"({delta_s_apex:.1f}m < -10.0m threshold), pinching the corner radius and choking exit speed."
                ),
            )
        )

    # Check 7: Exit Traction (|delta| > 25 deg when throttle crosses 50%)
    is_exit_lock = bool(u_exit_steer > 25.0)
    if is_exit_lock:
        flaws.append(
            DetectedFlaw(
                code="premature_throttle_lock",
                label="premature throttle application with excessive lock",
                phase="exit",
                symptom="oversteer",
                is_driver_error=True,
                metric_value=round(u_exit_steer, 2),
                reference_value=round(r_exit_steer, 2),
                threshold=25.0,
                explanation=(
                    f"Throttle crossed 50% while steering angle was still {u_exit_steer:.1f} deg "
                    f"(> 25.0 deg threshold, reference {r_exit_steer:.1f} deg), overloading lateral+longitudinal traction."
                ),
            )
        )

    # Check 9a: Early Braking / Overslowing (braking > 12m early AND entry time loss > 0.04s AND slower v_min)
    is_early_braking = bool(
        has_braking_zone and brake_point_offset_m < -12.0 and time_loss_entry_s > 0.04 and delta_v_min < -1.0 and not is_lazy_brake
    )
    if is_early_braking:
        flaws.append(
            DetectedFlaw(
                code="early_braking_overslow",
                label="early braking and entry overslow",
                phase="entry",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(brake_point_offset_m, 2),
                reference_value=0.0,
                threshold=-12.0,
                explanation=(
                    f"Braking started at {u_brake_onset_s:.1f}m vs reference {r_brake_onset_s:.1f}m "
                    f"({brake_point_offset_m:.1f}m early), losing {time_loss_entry_s:.2f}s on entry and overslowing the apex by {abs(delta_v_min):.1f} m/s."
                ),
            )
        )

    # Check 9b: Late Braking / Entry Overshoot (braking > 10m late AND apex speed slower by > 1.5 m/s)
    is_late_braking = bool(has_braking_zone and brake_point_offset_m > 10.0 and delta_v_min < -1.5)
    if is_late_braking:
        flaws.append(
            DetectedFlaw(
                code="late_braking_overshoot",
                label="late braking entry overshoot",
                phase="entry",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(brake_point_offset_m, 2),
                reference_value=0.0,
                threshold=10.0,
                explanation=(
                    f"Braking started +{brake_point_offset_m:.1f}m later than reference ({u_brake_onset_s:.1f}m vs {r_brake_onset_s:.1f}m), "
                    f"compromising rotation and costing {abs(delta_v_min):.1f} m/s at the apex."
                ),
            )
        )

    # Check 9c: Missing Trail-Brake Overlap (reference has significant trail-braking >= 15 G*m, user < 35% of ref)
    is_missing_trail = bool(r_trail_overlap >= 15.0 and u_trail_overlap < 0.35 * r_trail_overlap)
    if is_missing_trail:
        flaws.append(
            DetectedFlaw(
                code="missing_trail_brake_overlap",
                label="insufficient trail-braking overlap into turn-in",
                phase="entry",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(u_trail_overlap, 2),
                reference_value=round(r_trail_overlap, 2),
                threshold=round(0.35 * r_trail_overlap, 2),
                explanation=(
                    f"Trail-brake overlap integral (Brake * |a_y|) was {u_trail_overlap:.1f} G*m vs reference "
                    f"{r_trail_overlap:.1f} G*m, releasing brake before turn-in and starving the front axle of vertical load."
                ),
            )
        )

    # Check 10: Steering Sawing / Mid-Corner Corrections (>= 4 reversals and >= ref + 3)
    is_steering_sawing = bool(u_reversals >= 4 and u_reversals >= r_reversals + 3)
    if is_steering_sawing:
        flaws.append(
            DetectedFlaw(
                code="steering_sawing",
                label="mid-corner steering sawing / micro-corrections",
                phase="mid",
                symptom="understeer",
                is_driver_error=True,
                metric_value=float(u_reversals),
                reference_value=float(r_reversals),
                threshold=float(max(4, r_reversals + 3)),
                explanation=(
                    f"Detected {u_reversals} steering direction reversals (RMS rate {u_steer_rms:.1f} deg/s) vs "
                    f"reference {r_reversals} reversals ({r_steer_rms:.1f} deg/s), unsettling the aero/mechanical platform."
                ),
            )
        )

    # Check 11: Throttle Pumping / Exit Hesitation (> 15% cumulative throttle drop after apex)
    is_throttle_pumping = bool(u_th_hesitation > 15.0 and u_th_hesitation > r_th_hesitation + 12.0)
    if is_throttle_pumping:
        flaws.append(
            DetectedFlaw(
                code="throttle_pumping",
                label="post-apex throttle pumping / hesitation",
                phase="exit",
                symptom="oversteer",
                is_driver_error=True,
                metric_value=round(u_th_hesitation, 1),
                reference_value=round(r_th_hesitation, 1),
                threshold=15.0,
                explanation=(
                    f"Cumulative post-apex throttle lift/hesitation reached {u_th_hesitation:.1f}% vs reference "
                    f"{r_th_hesitation:.1f}%, indicating premature throttle commit before rotation was complete."
                ),
            )
        )

    # Check 12: Snap Oversteer / Counter-Steer Correction (>= 6m opposite lock under lateral load)
    is_countersteer = bool(countersteer_m >= 6.0)
    if is_countersteer:
        flaws.append(
            DetectedFlaw(
                code="snap_oversteer_countersteer",
                label="snap oversteer requiring counter-steer correction",
                phase="exit",
                symptom="oversteer",
                is_driver_error=True,
                metric_value=round(countersteer_m, 1),
                reference_value=0.0,
                threshold=6.0,
                explanation=(
                    f"Driver applied opposite-lock counter-steering across {countersteer_m:.1f}m while lateral load "
                    "exceeded 0.40G, indicating rear traction loss."
                ),
            )
        )

    # Check 14: Over-Charged Entry / Compromised Exit (v_min > ref + 1.0 m/s but v_exit < ref - 1.5 m/s and WOT delayed > 10m)
    is_exit_compromise = bool(delta_v_min > 1.0 and delta_v_exit < -1.5 and wot_offset_m > 10.0)
    if is_exit_compromise:
        flaws.append(
            DetectedFlaw(
                code="overslow_exit_compromise",
                label="over-charged apex speed compromising exit acceleration",
                phase="exit",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(delta_v_exit, 2),
                reference_value=0.0,
                threshold=-1.5,
                explanation=(
                    f"Carried +{delta_v_min:.1f} m/s higher minimum speed at the apex but delayed full throttle by "
                    f"+{wot_offset_m:.1f}m and exited {abs(delta_v_exit):.1f} m/s slower than reference."
                ),
            )
        )

    # Check 15a: Excessive ABS Intervention (> 15m in ABS and > ref + 10m)
    is_excessive_abs = bool(u_abs_m > 15.0 and u_abs_m > r_abs_m + 10.0)
    if is_excessive_abs:
        flaws.append(
            DetectedFlaw(
                code="excessive_abs_intervention",
                label="excessive ABS activation overheating front tires",
                phase="entry",
                symptom="understeer",
                is_driver_error=True,
                metric_value=round(u_abs_m, 1),
                reference_value=round(r_abs_m, 1),
                threshold=round(max(15.0, r_abs_m + 10.0), 1),
                explanation=(
                    f"ABS remained active for {u_abs_m:.1f}m vs reference {r_abs_m:.1f}m, "
                    "exceeding peak longitudinal slip ratio and saturating front tire surface temperatures."
                ),
            )
        )

    # Check 15b: Late Mid-Corner Downshift (>= 1 downshift well after turn-in and > ref)
    is_late_downshift = bool(u_late_shifts >= 1 and u_late_shifts > 0 and u_min_gear > 0)
    if is_late_downshift:
        flaws.append(
            DetectedFlaw(
                code="late_mid_corner_downshift",
                label="late mid-corner downshift unsettling rear axle",
                phase="mid",
                symptom="oversteer",
                is_driver_error=True,
                metric_value=float(u_late_shifts),
                reference_value=0.0,
                threshold=1.0,
                explanation=(
                    f"Executed {u_late_shifts} downshift(s) after turn-in while loaded laterally (minimum gear {u_min_gear}), "
                    "spiking rear engine-braking torque mid-corner."
                ),
            )
        )

    technique_fault_isolated = any(f.is_driver_error for f in flaws)

    return CornerAnalysisResult(
        corner_number=segment.corner_number,
        corner_name=segment.name,
        start_m=segment.start_m,
        end_m=segment.end_m,
        user_brake_rise_time_s=round(u_rise, 4),
        ref_brake_rise_time_s=round(r_rise, 4),
        brake_rise_delta_s=round(rise_delta, 4),
        user_trail_off_gradient_pct_per_m=round(u_trail_grad, 3),
        ref_trail_off_gradient_pct_per_m=round(r_trail_grad, 3),
        user_coasting_distance_m=round(u_coast_m, 2),
        ref_coasting_distance_m=round(r_coast_m, 2),
        coasting_delta_m=round(coast_delta_m, 2),
        user_min_g_sum_transition=round(u_min_g, 3),
        ref_min_g_sum_transition=round(r_min_g, 3),
        steering_scrub_delta_deg=round(scrub_delta_deg, 2),
        user_v_min=round(u_v_min, 3),
        ref_v_min=round(r_v_min, 3),
        delta_v_min=round(delta_v_min, 3),
        user_apex_s_m=round(u_apex_s, 2),
        ref_apex_s_m=round(r_apex_s, 2),
        delta_s_apex_m=round(delta_s_apex, 2),
        user_exit_steering_at_50pct_throttle_deg=round(u_exit_steer, 2),
        ref_exit_steering_at_50pct_throttle_deg=round(r_exit_steer, 2),
        lazy_brake_hit=is_lazy_brake,
        trail_off_warning=is_trail_off_dump,
        dead_time_coasting=is_coasting,
        g_sum_sag=is_g_sag,
        steering_scrub=is_steering_scrub,
        early_apex_pinch=is_early_apex,
        premature_throttle_lock=is_exit_lock,
        time_loss_total_s=round(time_loss_total_s, 4),
        time_loss_entry_s=round(time_loss_entry_s, 4),
        time_loss_mid_s=round(time_loss_mid_s, 4),
        time_loss_exit_s=round(time_loss_exit_s, 4),
        time_loss_straight_s=round(time_loss_straight_s, 4),
        user_brake_onset_s_m=round(u_brake_onset_s, 2),
        ref_brake_onset_s_m=round(r_brake_onset_s, 2),
        brake_point_offset_m=round(brake_point_offset_m, 2),
        user_trail_brake_overlap_gs_m=round(u_trail_overlap, 2),
        ref_trail_brake_overlap_gs_m=round(r_trail_overlap, 2),
        user_steering_rate_rms_deg_per_s=round(u_steer_rms, 2),
        ref_steering_rate_rms_deg_per_s=round(r_steer_rms, 2),
        user_steering_reversals=u_reversals,
        ref_steering_reversals=r_reversals,
        user_throttle_hesitation_pct=round(u_th_hesitation, 2),
        ref_throttle_hesitation_pct=round(r_th_hesitation, 2),
        understeer_index_delta=round(understeer_index_delta, 3),
        countersteer_distance_m=round(countersteer_m, 2),
        user_grip_utilization_pct=round(u_grip_pct, 2),
        ref_grip_utilization_pct=round(r_grip_pct, 2),
        grip_utilization_delta_pct=round(grip_delta_pct, 2),
        user_v_exit=round(u_v_exit, 3),
        ref_v_exit=round(r_v_exit, 3),
        delta_v_exit=round(delta_v_exit, 3),
        full_throttle_offset_m=round(wot_offset_m, 2),
        corner_line_style=corner_line_style,
        user_abs_active_distance_m=round(u_abs_m, 2),
        ref_abs_active_distance_m=round(r_abs_m, 2),
        user_min_gear=u_min_gear,
        ref_min_gear=r_min_gear,
        user_late_downshifts_after_turn_in=u_late_shifts,
        early_braking_overslow=is_early_braking,
        late_braking_overshoot=is_late_braking,
        missing_trail_brake_overlap=is_missing_trail,
        steering_sawing=is_steering_sawing,
        throttle_pumping=is_throttle_pumping,
        snap_oversteer_countersteer=is_countersteer,
        overslow_exit_compromise=is_exit_compromise,
        excessive_abs_intervention=is_excessive_abs,
        late_mid_corner_downshift=is_late_downshift,
        technique_fault_isolated=technique_fault_isolated,
        flaws=[f.to_dict() for f in flaws],
    )


def analyze_telemetry(
    user_lap: Any,
    ref_lap: Any,
    track_slug: str | None = None,
    specific_corner: int | None = None,
    segments: list[CornerSegment] | None = None,
) -> dict[str, Any]:
    """Run full spatial alignment and deterministic corner/sector physics analysis.

    Returns a dictionary (serializable from `LapAnalysisReport`) containing:
    - `technique_fault_isolated`: bool
    - `flaws`: list[dict]
    - `corners`: list[dict]
    - `summary_deltas`: dict[str, Any]
    - `top_time_loss_corners`: list[dict]
    """
    u_aligned, r_aligned = align_laps_spatially(user_lap, ref_lap, step_m=1.0)
    s_grid = u_aligned["distance_m"]

    # Estimate 98th-percentile tire friction ellipse envelope (G_x,max, G_y,max) from reference lap
    gx_max = float(max(np.percentile(np.abs(r_aligned["lon_accel_g"]), 98), 1.0))
    gy_max = float(max(np.percentile(np.abs(r_aligned["lat_accel_g"]), 98), 1.0))

    if segments is None:
        segments = get_track_corners(
            track_slug=track_slug,
            distance_m=s_grid,
            lat_accel_g=r_aligned["lat_accel_g"],
            speed=r_aligned["speed"],
        )

    if specific_corner is not None:
        selected = [seg for seg in segments if seg.corner_number == specific_corner]
        if selected:
            segments = selected
        elif 1 <= specific_corner <= len(segments):
            segments = [segments[specific_corner - 1]]

    if not segments:
        segments = [
            CornerSegment(
                corner_number=1,
                name="Corner 1",
                start_m=float(s_grid[0]),
                end_m=float(s_grid[-1]),
                nominal_apex_m=float(s_grid[int(np.argmin(r_aligned["speed"]))]),
            )
        ]

    corner_results: list[CornerAnalysisResult] = []
    all_flaws: list[dict[str, Any]] = []

    for idx, seg in enumerate(segments):
        next_start = segments[idx + 1].start_m if idx + 1 < len(segments) else float(s_grid[-1])
        res = analyze_corner_segment(
            u_aligned,
            r_aligned,
            seg,
            next_segment_start_m=next_start,
            gx_max=gx_max,
            gy_max=gy_max,
        )
        corner_results.append(res)
        for flaw in res.flaws:
            enriched = dict(flaw)
            enriched["corner_number"] = res.corner_number
            enriched["corner_name"] = res.corner_name
            all_flaws.append(enriched)

    technique_fault_isolated = any(c.technique_fault_isolated for c in corner_results)

    # Rank corners by total time lost (corner + following straight carry)
    ranked_corners = sorted(
        corner_results,
        key=lambda c: c.time_loss_total_s + c.time_loss_straight_s,
        reverse=True,
    )
    top_time_loss_corners = [
        {
            "corner_number": c.corner_number,
            "corner_name": c.corner_name,
            "time_loss_corner_s": c.time_loss_total_s,
            "time_loss_straight_s": c.time_loss_straight_s,
            "combined_time_loss_s": round(c.time_loss_total_s + c.time_loss_straight_s, 4),
            "time_loss_entry_s": c.time_loss_entry_s,
            "time_loss_mid_s": c.time_loss_mid_s,
            "time_loss_exit_s": c.time_loss_exit_s,
        }
        for c in ranked_corners[:3]
    ]

    summary_deltas: dict[str, Any] = {
        "max_brake_rise_delta_s": round(max((c.brake_rise_delta_s for c in corner_results), default=0.0), 4),
        "max_trail_off_gradient_pct_per_m": round(
            max((c.user_trail_off_gradient_pct_per_m for c in corner_results), default=0.0), 3
        ),
        "max_coasting_delta_m": round(max((c.coasting_delta_m for c in corner_results), default=0.0), 2),
        "min_g_sum_transition": round(min((c.user_min_g_sum_transition for c in corner_results), default=0.0), 3),
        "max_steering_scrub_delta_deg": round(
            max((c.steering_scrub_delta_deg for c in corner_results), default=0.0), 2
        ),
        "min_delta_v_min": round(min((c.delta_v_min for c in corner_results), default=0.0), 3),
        "min_delta_s_apex_m": round(min((c.delta_s_apex_m for c in corner_results), default=0.0), 2),
        "max_exit_steering_at_50pct_throttle_deg": round(
            max((c.user_exit_steering_at_50pct_throttle_deg for c in corner_results), default=0.0), 2
        ),
        "total_corner_time_loss_s": round(sum(c.time_loss_total_s for c in corner_results), 4),
        "total_straight_carry_time_loss_s": round(sum(c.time_loss_straight_s for c in corner_results), 4),
        "mean_user_grip_utilization_pct": round(
            float(np.mean([c.user_grip_utilization_pct for c in corner_results])) if corner_results else 0.0, 2
        ),
        "mean_ref_grip_utilization_pct": round(
            float(np.mean([c.ref_grip_utilization_pct for c in corner_results])) if corner_results else 0.0, 2
        ),
        "max_steering_reversals": max((c.user_steering_reversals for c in corner_results), default=0),
        "max_throttle_hesitation_pct": round(
            max((c.user_throttle_hesitation_pct for c in corner_results), default=0.0), 2
        ),
        "max_countersteer_distance_m": round(
            max((c.countersteer_distance_m for c in corner_results), default=0.0), 2
        ),
        "max_abs_active_distance_m": round(
            max((c.user_abs_active_distance_m for c in corner_results), default=0.0), 2
        ),
        "total_late_downshifts": sum(c.user_late_downshifts_after_turn_in for c in corner_results),
    }

    report = LapAnalysisReport(
        track_length_m=float(s_grid[-1] - s_grid[0]),
        technique_fault_isolated=technique_fault_isolated,
        flaws=all_flaws,
        corners=[c.to_dict() for c in corner_results],
        summary_deltas=summary_deltas,
        top_time_loss_corners=top_time_loss_corners,
    )
    return report.to_dict()

