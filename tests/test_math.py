"""Synthetic telemetry physics unit tests for `telemetry/signal_math.py` and `telemetry/track_sectors.py`."""

from __future__ import annotations

import numpy as np

from telemetry.signal_math import align_laps_spatially, analyze_telemetry
from telemetry.track_sectors import CornerSegment, auto_segment_corners, get_track_corners


def make_reference_corner_lap(num_points: int = 301) -> dict[str, np.ndarray]:
    """Create a realistic, clean reference corner telemetry trace from 0m to 300m.

    - Braking starts sharply at s=60m, rises from 0 to 0.90 in 0.06s (2m),
      then trails off smoothly at ~1.5%/m to the apex at s=150m.
    - Apex minimum speed v_min = 35.0 m/s at s=150m.
    - Steering rises smoothly to 30 deg at the apex and unwinds to 12 deg before 50% throttle.
    - Combined G-sum remains >= 1.25G throughout the pedal transition.
    """
    s = np.linspace(0.0, 300.0, num_points)
    # Time axis at roughly 40 m/s average speed
    t = s / 40.0

    # Speed profile: 65 m/s entry, 35 m/s minimum at s=150m, 60 m/s exit
    speed = 35.0 + 30.0 * np.square((s - 150.0) / 150.0)

    brake = np.zeros_like(s)
    # Sharp brake hit at s=60..62m (0.05s rise time), smooth trail-off from s=62..150m (~1.0%/m)
    brake[(s >= 60.0) & (s < 62.0)] = np.linspace(0.05, 0.90, np.sum((s >= 60.0) & (s < 62.0)))
    brake[(s >= 62.0) & (s <= 150.0)] = np.linspace(0.90, 0.06, np.sum((s >= 62.0) & (s <= 150.0)))

    throttle = np.zeros_like(s)
    throttle[s < 58.0] = 1.0
    # Immediate smooth throttle pickup right after apex (s=151m..220m)
    throttle[(s >= 151.0) & (s <= 220.0)] = np.linspace(0.08, 1.0, np.sum((s >= 151.0) & (s <= 220.0)))
    throttle[s > 220.0] = 1.0

    # Steering angle: turn-in at s=80m, peaks at 30 deg around apex (s=150m), unwinds to <15 deg by s=180m
    steering_deg = 30.0 * np.exp(-0.5 * np.square((s - 145.0) / 30.0))
    steering_deg[s < 75.0] = 0.0

    # Lateral and longitudinal Gs maintaining >= 1.2G friction circle during transition
    lat_accel_g = 1.45 * np.exp(-0.5 * np.square((s - 140.0) / 55.0))
    lon_accel_g = -1.45 * brake + 0.45 * throttle

    return {
        "distance_m": s,
        "time_s": t,
        "speed": speed,
        "brake": brake,
        "throttle": throttle,
        "steering_deg": steering_deg,
        "lat_accel_g": lat_accel_g,
        "lon_accel_g": lon_accel_g,
    }


SINGLE_CORNER = [CornerSegment(corner_number=1, name="Turn 1", start_m=0.0, end_m=300.0, nominal_apex_m=150.0)]


def test_spatial_resampling_1m_grid() -> None:
    """Verify spatial alignment interpolates irregular samples onto an exact 1.0m distance axis."""
    ref = make_reference_corner_lap(num_points=180)
    user = make_reference_corner_lap(num_points=240)

    u_aligned, r_aligned = align_laps_spatially(user, ref, step_m=1.0)

    assert np.allclose(np.diff(u_aligned["distance_m"]), 1.0)
    assert np.array_equal(u_aligned["distance_m"], r_aligned["distance_m"])
    assert u_aligned["distance_m"][0] == 0.0
    assert u_aligned["distance_m"][-1] == 300.0


def test_lazy_brake_hit_triggered() -> None:
    """Verify that early braking + slow brake pressure rise time (> ref + 0.10s) triggers `lazy_brake_hit`."""
    ref = make_reference_corner_lap()
    user = make_reference_corner_lap()

    s = user["distance_m"]
    user["brake"] = np.zeros_like(s)
    # User starts braking earlier at s=45m and takes 20 meters (~0.50s) to ramp from 5% to 85% pressure
    ramp_mask = (s >= 45.0) & (s <= 65.0)
    user["brake"][ramp_mask] = np.linspace(0.05, 0.85, np.sum(ramp_mask))
    trail_mask = (s > 65.0) & (s <= 150.0)
    user["brake"][trail_mask] = np.linspace(0.85, 0.06, np.sum(trail_mask))

    report = analyze_telemetry(user, ref, segments=SINGLE_CORNER)
    corner = report["corners"][0]

    assert corner["lazy_brake_hit"] is True
    assert corner["brake_rise_delta_s"] > 0.10
    assert report["technique_fault_isolated"] is True
    flaw_codes = [f["code"] for f in report["flaws"]]
    flaw_labels = [f["label"] for f in report["flaws"]]
    assert "lazy_brake_hit" in flaw_codes
    assert "lazy brake hit" in flaw_labels


def test_steering_scrub_triggered() -> None:
    """Verify that excessive steering angle (> +12 deg) with negative apex speed delta triggers `steering_scrub`."""
    ref = make_reference_corner_lap()
    user = make_reference_corner_lap()

    s = user["distance_m"]
    # Add +18 deg excess steering lock from turn-in (s=80m) to apex (s=150m)
    turn_in_to_apex = (s >= 75.0) & (s <= 160.0)
    user["steering_deg"][turn_in_to_apex] += 18.0
    # Lower user apex minimum speed by 3.5 m/s relative to reference
    user["speed"] = user["speed"] - 3.5 * np.exp(-0.5 * np.square((s - 150.0) / 25.0))

    report = analyze_telemetry(user, ref, segments=SINGLE_CORNER)
    corner = report["corners"][0]

    assert corner["steering_scrub"] is True
    assert corner["steering_scrub_delta_deg"] > 12.0
    assert corner["delta_v_min"] < 0.0
    assert report["technique_fault_isolated"] is True
    flaw_codes = [f["code"] for f in report["flaws"]]
    assert "steering_scrub" in flaw_codes


def test_abrupt_brake_release_trail_off_warning() -> None:
    """Verify that dumping the brake pedal (> 10%/m release rate) triggers the trail-off warning."""
    ref = make_reference_corner_lap()
    user = make_reference_corner_lap()

    s = user["distance_m"]
    # User hits 90% brake sharply at s=60..62m, holds until s=95m, then dumps from 85% to 0% in 4 meters (21.25%/m)
    user["brake"] = np.zeros_like(s)
    user["brake"][(s >= 60.0) & (s < 62.0)] = np.linspace(0.05, 0.90, np.sum((s >= 60.0) & (s < 62.0)))
    user["brake"][(s >= 62.0) & (s <= 95.0)] = 0.85
    user["brake"][(s > 95.0) & (s <= 99.0)] = np.linspace(0.70, 0.0, np.sum((s > 95.0) & (s <= 99.0)))
    # Keep tiny drag brake after dump so we isolate trail-off warning
    user["brake"][(s > 99.0) & (s <= 150.0)] = 0.06

    report = analyze_telemetry(user, ref, segments=SINGLE_CORNER)
    corner = report["corners"][0]

    assert corner["trail_off_warning"] is True
    assert corner["user_trail_off_gradient_pct_per_m"] > 10.0
    assert report["technique_fault_isolated"] is True
    flaw_labels = [f["label"] for f in report["flaws"]]
    assert "abrupt brake release unweighting front axle" in flaw_labels


def test_coasting_g_sum_early_apex_and_exit_traction_flags() -> None:
    """Verify dead-time coasting, G-sum sag, early apex pinch, and premature exit throttle with lock."""
    ref = make_reference_corner_lap()
    user = make_reference_corner_lap()

    s = user["distance_m"]
    # 1. Early apex pinch: shift minimum speed to s=132m (-18m before reference apex at 150m)
    user["speed"] = 33.0 + 30.0 * np.square((s - 132.0) / 150.0)

    # 2. Coasting > 8m: zero both brake and throttle between s=105m and s=125m (21m contiguous coasting)
    coast_zone = (s >= 105.0) & (s <= 125.0)
    user["brake"][coast_zone] = 0.0
    user["throttle"][coast_zone] = 0.0

    # 3. G-sum sag < 1.0G during transition
    user["lat_accel_g"][coast_zone] = 0.45
    user["lon_accel_g"][coast_zone] = -0.10

    # 4. Premature throttle with > 25 deg steering lock on exit
    exit_zone = (s >= 132.0) & (s <= 170.0)
    user["throttle"][exit_zone] = 0.65
    user["steering_deg"][exit_zone] = 32.0

    report = analyze_telemetry(user, ref, segments=SINGLE_CORNER)
    corner = report["corners"][0]

    assert corner["dead_time_coasting"] is True
    assert corner["g_sum_sag"] is True
    assert corner["early_apex_pinch"] is True
    assert corner["premature_throttle_lock"] is True


def test_clean_lap_has_zero_flaws() -> None:
    """Verify that a clean user lap matching reference technique produces `technique_fault_isolated: False`."""
    ref = make_reference_corner_lap()
    user = make_reference_corner_lap()
    # Slight uniform speed offset without technique flaws
    user["speed"] = user["speed"] - 0.3

    report = analyze_telemetry(user, ref, segments=SINGLE_CORNER)
    assert report["technique_fault_isolated"] is False
    assert report["flaws"] == []


def test_track_sector_premapped_and_auto_segmentation() -> None:
    """Verify pre-mapped tracks and fallback |a_y| > 0.5G auto-segmentation."""
    spa_corners = get_track_corners("spa")
    assert len(spa_corners) >= 10
    assert spa_corners[0].name == "La Source"

    ref = make_reference_corner_lap()
    auto_corners = auto_segment_corners(ref["distance_m"], ref["lat_accel_g"], ref["speed"])
    assert len(auto_corners) == 1
    assert abs(auto_corners[0].nominal_apex_m - 150.0) <= 2.0


def test_phase_time_loss_and_braking_point_offset_and_trail_overlap() -> None:
    """Verify spatial phase time loss decomposition, early/late braking flags, and trail-brake overlap integral."""
    ref = make_reference_corner_lap()
    user_early = make_reference_corner_lap()

    s = user_early["distance_m"]
    # User brakes 18m early at s=42m (with sharp initial rise so lazy_brake_hit is not triggered),
    # releases brake completely before turn-in (s=75m), and runs 4 m/s slower through entry & apex
    user_early["brake"] = np.zeros_like(s)
    user_early["brake"][(s >= 42.0) & (s < 44.0)] = np.linspace(0.05, 0.90, np.sum((s >= 42.0) & (s < 44.0)))
    user_early["brake"][(s >= 44.0) & (s <= 72.0)] = np.linspace(0.90, 0.02, np.sum((s >= 44.0) & (s <= 72.0)))
    user_early["speed"] = user_early["speed"] - 4.0

    report_early = analyze_telemetry(user_early, ref, segments=SINGLE_CORNER)
    c_early = report_early["corners"][0]

    assert c_early["brake_point_offset_m"] <= -15.0
    assert c_early["early_braking_overslow"] is True
    assert c_early["missing_trail_brake_overlap"] is True
    assert c_early["time_loss_total_s"] > 0.20
    assert len(report_early["top_time_loss_corners"]) == 1

    # Late braking overshoot test
    user_late = make_reference_corner_lap()
    user_late["brake"] = np.zeros_like(s)
    user_late["brake"][(s >= 74.0) & (s < 76.0)] = np.linspace(0.05, 0.90, np.sum((s >= 74.0) & (s < 76.0)))
    user_late["brake"][(s >= 76.0) & (s <= 150.0)] = np.linspace(0.90, 0.10, np.sum((s >= 76.0) & (s <= 150.0)))
    user_late["speed"][(s >= 120.0) & (s <= 180.0)] -= 3.0

    report_late = analyze_telemetry(user_late, ref, segments=SINGLE_CORNER)
    c_late = report_late["corners"][0]
    assert c_late["brake_point_offset_m"] >= 12.0
    assert c_late["late_braking_overshoot"] is True


def test_steering_sawing_throttle_pumping_and_countersteer() -> None:
    """Verify mid-corner steering sawing reversals, post-apex throttle pumping, and counter-steer detection."""
    ref = make_reference_corner_lap()
    user = make_reference_corner_lap()
    s = user["distance_m"]

    # 1. Inject sinusoidal sawing (+-4 deg oscillation every 6m) between turn-in (s=80m) and apex (s=150m)
    mid_mask = (s >= 80.0) & (s <= 150.0)
    user["steering_deg"][mid_mask] += 4.0 * np.sin((s[mid_mask] - 80.0) * (2.0 * np.pi / 12.0))

    # 2. Inject post-apex throttle pumping (ramp to 60%, drop to 20%, ramp to 100%)
    user["throttle"][(s >= 151.0) & (s <= 165.0)] = np.linspace(0.10, 0.60, np.sum((s >= 151.0) & (s <= 165.0)))
    user["throttle"][(s > 165.0) & (s <= 175.0)] = np.linspace(0.55, 0.20, np.sum((s > 165.0) & (s <= 175.0)))
    user["throttle"][(s > 175.0) & (s <= 215.0)] = np.linspace(0.25, 1.0, np.sum((s > 175.0) & (s <= 215.0)))

    # 3. Inject 10 meters of opposite-lock counter-steering on exit (s=175..185m) while lat_accel_g > 0.5G
    cs_mask = (s >= 175.0) & (s <= 185.0)
    user["steering_deg"][cs_mask] = -18.0
    user["lat_accel_g"][cs_mask] = 0.85

    report = analyze_telemetry(user, ref, segments=SINGLE_CORNER)
    corner = report["corners"][0]

    assert corner["steering_sawing"] is True
    assert corner["user_steering_reversals"] >= 4
    assert corner["throttle_pumping"] is True
    assert corner["user_throttle_hesitation_pct"] > 30.0
    assert corner["snap_oversteer_countersteer"] is True
    assert corner["countersteer_distance_m"] >= 10.0


def test_v_vs_u_exit_compromise_grip_utilization_and_optional_channels() -> None:
    """Verify over-charged apex exit compromise, grip utilization %, ABS intervention, and late downshifts."""
    ref = make_reference_corner_lap()
    user = make_reference_corner_lap()
    s = user["distance_m"]

    # Add optional channels to both ref and user
    ref["abs_active"] = np.zeros_like(s)
    ref["gear"] = np.where(s < 85.0, 4, 2)

    user["abs_active"] = np.zeros_like(s)
    # User triggers ABS for 25 meters during braking (s=65..90m)
    user["abs_active"][(s >= 65.0) & (s <= 90.0)] = 1.0

    # User downshifts late: 4 -> 3 at s=75m, then 3 -> 2 at s=110m (30m after turn-in)
    user["gear"] = np.full_like(s, 4.0)
    user["gear"][(s >= 75.0) & (s < 110.0)] = 3.0
    user["gear"][s >= 110.0] = 2.0

    # Over-charged entry / compromised exit: user carries +2.0 m/s higher minimum speed at apex (s=150m),
    # delays full throttle by +25m, and exits 3.0 m/s slower at s=300m
    user["speed"] = 37.0 + 28.0 * np.square((s - 150.0) / 150.0)
    user["speed"][s >= 180.0] -= 4.5 * ((s[s >= 180.0] - 180.0) / 120.0)
    user["throttle"][(s >= 151.0) & (s <= 250.0)] = np.linspace(0.08, 0.95, np.sum((s >= 151.0) & (s <= 250.0)))

    report = analyze_telemetry(user, ref, segments=SINGLE_CORNER)
    corner = report["corners"][0]

    assert corner["overslow_exit_compromise"] is True
    assert corner["corner_line_style"] == "overcharged_entry_compromised_exit"
    assert corner["excessive_abs_intervention"] is True
    assert corner["user_abs_active_distance_m"] >= 25.0
    assert corner["late_mid_corner_downshift"] is True
    assert corner["user_late_downshifts_after_turn_in"] >= 1
    assert corner["user_grip_utilization_pct"] > 0.0
    assert corner["ref_grip_utilization_pct"] > 0.0

