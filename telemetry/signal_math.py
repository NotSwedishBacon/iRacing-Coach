"""Deterministic NumPy/SciPy Telemetry Analysis Engine (`telemetry/signal_math.py`).

Implements spatial distance alignment onto a common 1.0-meter distance axis
s in [0, TrackLength] via `scipy.interpolate.interp1d` and deterministic physics
diagnostics per corner/sector to isolate driver technique errors vs. vehicle balance flaws:

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
    lazy_brake_hit: bool = False
    trail_off_warning: bool = False
    dead_time_coasting: bool = False
    g_sum_sag: bool = False
    steering_scrub: bool = False
    early_apex_pinch: bool = False
    premature_throttle_lock: bool = False
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
    summary_deltas: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_channel_dict(lap_data: Any) -> dict[str, np.ndarray]:
    """Convert a pandas DataFrame or dict of arrays into normalized 1D NumPy arrays."""
    if hasattr(lap_data, "to_dict") and hasattr(lap_data, "columns"):
        from telemetry.g61_client import normalize_telemetry_columns

        df = normalize_telemetry_columns(lap_data)
        raw: Mapping[str, Any] = {col: df[col].to_numpy(dtype=float) for col in df.columns}
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

    return {
        "distance_m": s,
        "time_s": time_s,
        "speed": speed,
        "brake": brake,
        "throttle": throttle,
        "steering_deg": steering_deg,
        "lat_accel_g": lat_accel_g,
        "lon_accel_g": lon_accel_g,
    }


def resample_lap_spatial(
    lap_data: Any,
    track_length_m: float | None = None,
    step_m: float = 1.0,
    s_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Interpolate a lap onto a uniform 1.0-meter spatial distance axis s in [0, TrackLength].

    Uses `scipy.interpolate.interp1d` with linear interpolation and boundary clamping.
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
        interpolator = interp1d(
            s_unique,
            v_unique,
            kind="linear",
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
    # Use the 90th percentile / mean peak excess steering lock across the turn-in-to-apex window
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


def analyze_corner_segment(
    user_aligned: Mapping[str, np.ndarray],
    ref_aligned: Mapping[str, np.ndarray],
    segment: CornerSegment,
) -> CornerAnalysisResult:
    """Run all 7 deterministic physics checks on a single corner segment."""
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
    - `summary_deltas`: dict[str, float]
    """
    u_aligned, r_aligned = align_laps_spatially(user_lap, ref_lap, step_m=1.0)
    s_grid = u_aligned["distance_m"]

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

    for seg in segments:
        res = analyze_corner_segment(u_aligned, r_aligned, seg)
        corner_results.append(res)
        for flaw in res.flaws:
            enriched = dict(flaw)
            enriched["corner_number"] = res.corner_number
            enriched["corner_name"] = res.corner_name
            all_flaws.append(enriched)

    technique_fault_isolated = any(c.technique_fault_isolated for c in corner_results)

    summary_deltas = {
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
    }

    report = LapAnalysisReport(
        track_length_m=float(s_grid[-1] - s_grid[0]),
        technique_fault_isolated=technique_fault_isolated,
        flaws=all_flaws,
        corners=[c.to_dict() for c in corner_results],
        summary_deltas=summary_deltas,
    )
    return report.to_dict()
