"""Track corner segment definitions and automatic lateral-G corner segmentation.

Includes pre-mapped corner distance ranges (in meters) for major iRacing circuits:
- Spa-Francorchamps (`spa`, `circuit-de-spa-francorchamps`)
- Nürburgring GP (`nurburgring-gp`, `nurburgring-grand-prix`)
- Watkins Glen International - Boot (`watkins-glen`, `watkins-glen-boot`)
- Road America (`road-america`)

When a track is not pre-mapped, `auto_segment_corners` detects corners where
|a_y| > 0.5 G and locates each corner's local minimum speed (v_min) to construct
entry, apex, and exit bounds automatically.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CornerSegment:
    """Spatial definition of a track corner along the 1.0m distance axis."""

    corner_number: int
    name: str
    start_m: float
    end_m: float
    nominal_apex_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PREMAPPED_TRACKS: dict[str, list[CornerSegment]] = {
    "spa": [
        CornerSegment(1, "La Source", 180.0, 420.0, 295.0),
        CornerSegment(2, "Eau Rouge / Raidillon", 780.0, 1220.0, 980.0),
        CornerSegment(3, "Les Combes (Right)", 1980.0, 2170.0, 2080.0),
        CornerSegment(4, "Les Combes (Left) / Malmedy", 2170.0, 2430.0, 2290.0),
        CornerSegment(5, "Bruxelles / Rivage", 2580.0, 2920.0, 2760.0),
        CornerSegment(6, "Speaker's Corner (No Name)", 3050.0, 3290.0, 3175.0),
        CornerSegment(7, "Pouhon", 3550.0, 4020.0, 3760.0),
        CornerSegment(8, "Les Fagnes (Campus)", 4250.0, 4630.0, 4440.0),
        CornerSegment(9, "Stavelot", 4720.0, 5120.0, 4910.0),
        CornerSegment(10, "Blanchimont", 5850.0, 6220.0, 6040.0),
        CornerSegment(11, "Bus Stop Chicane", 6480.0, 6860.0, 6650.0),
    ],
    "nurburgring-gp": [
        CornerSegment(1, "Yokohama-S (T1 Hairpin)", 520.0, 820.0, 675.0),
        CornerSegment(2, "Mercedes Arena Entry", 830.0, 1080.0, 950.0),
        CornerSegment(3, "Mercedes Arena Exit (T3/T4)", 1080.0, 1380.0, 1210.0),
        CornerSegment(4, "Ford-Kurve", 1520.0, 1790.0, 1660.0),
        CornerSegment(5, "Dunlop-Kehre", 1980.0, 2320.0, 2150.0),
        CornerSegment(6, "Schumacher-S", 2560.0, 2900.0, 2730.0),
        CornerSegment(7, "RTL-Kurve / Warsteiner", 3040.0, 3360.0, 3190.0),
        CornerSegment(8, "ADVAN-Bogen", 3580.0, 3920.0, 3750.0),
        CornerSegment(9, "NGK Chicane", 4420.0, 4740.0, 4580.0),
        CornerSegment(10, "Coca-Cola Kurve (Final Turn)", 4800.0, 5080.0, 4940.0),
    ],
    "watkins-glen": [
        CornerSegment(1, "Turn 1 (The Ninety)", 520.0, 860.0, 690.0),
        CornerSegment(2, "The Esses", 1050.0, 1650.0, 1350.0),
        CornerSegment(3, "Inner Loop (Bus Stop)", 2420.0, 2760.0, 2590.0),
        CornerSegment(4, "Outer Loop (Carousel)", 2820.0, 3260.0, 3040.0),
        CornerSegment(5, "The Chute (T6)", 3450.0, 3780.0, 3610.0),
        CornerSegment(6, "The Toe of the Boot (T7)", 3920.0, 4280.0, 4100.0),
        CornerSegment(7, "The Heel of the Boot (T8)", 4420.0, 4720.0, 4570.0),
        CornerSegment(8, "Off-Camber Left (T9)", 4780.0, 5040.0, 4910.0),
        CornerSegment(9, "Turn 10", 5140.0, 5360.0, 5250.0),
        CornerSegment(10, "Turn 11 (Final Corner)", 5380.0, 5580.0, 5480.0),
    ],
    "road-america": [
        CornerSegment(1, "Turn 1", 780.0, 1140.0, 960.0),
        CornerSegment(2, "Turn 3", 1420.0, 1760.0, 1590.0),
        CornerSegment(3, "Turn 5 (Moraine Sweep End)", 2560.0, 2920.0, 2740.0),
        CornerSegment(4, "Turn 6 (Toyota Bridge)", 3050.0, 3320.0, 3185.0),
        CornerSegment(5, "Turn 7 (Hurry Downs)", 3360.0, 3640.0, 3500.0),
        CornerSegment(6, "Turn 8", 3860.0, 4160.0, 4010.0),
        CornerSegment(7, "Carousel (T9-T10)", 4220.0, 4760.0, 4490.0),
        CornerSegment(8, "The Kink (T11)", 4920.0, 5180.0, 5050.0),
        CornerSegment(9, "Canada Corner (T12)", 5620.0, 5960.0, 5790.0),
        CornerSegment(10, "Turn 13 (Bill Mitchell Bend)", 6020.0, 6260.0, 6140.0),
        CornerSegment(11, "Turn 14 (Final Corner)", 6320.0, 6580.0, 6450.0),
    ],
}

TRACK_SLUG_ALIASES: dict[str, str] = {
    "spa": "spa",
    "spa-francorchamps": "spa",
    "circuit-de-spa-francorchamps": "spa",
    "nurburgring": "nurburgring-gp",
    "nurburgring-gp": "nurburgring-gp",
    "nurburgring-grand-prix": "nurburgring-gp",
    "watkins-glen": "watkins-glen",
    "watkins-glen-boot": "watkins-glen",
    "watkins-glen-international": "watkins-glen",
    "road-america": "road-america",
    "roadamerica": "road-america",
}


def auto_segment_corners(
    distance_m: np.ndarray,
    lat_accel_g: np.ndarray,
    speed: np.ndarray,
    lat_g_threshold: float = 0.5,
    min_corner_length_m: float = 25.0,
    entry_margin_m: float = 60.0,
    exit_margin_m: float = 40.0,
) -> list[CornerSegment]:
    """Automatically segment corners where |a_y| > 0.5G and locate local minimum speed v_min.

    Expands each detected corner window backward by `entry_margin_m` (to capture the
    braking hit and initial turn-in) and forward by `exit_margin_m` (to capture exit
    throttle application).
    """
    s = np.asarray(distance_m, dtype=float)
    ay = np.abs(np.asarray(lat_accel_g, dtype=float))
    v = np.asarray(speed, dtype=float)

    if s.size == 0:
        return []

    in_corner = ay > lat_g_threshold
    # Bridge brief 1-5m dips in |a_y| inside the same corner
    padded = np.concatenate(([False], in_corner, [False]))
    starts = np.where(padded[1:] & ~padded[:-1])[0]
    ends = np.where(~padded[1:] & padded[:-1])[0] - 1

    if len(starts) == 0:
        # Fallback single segment covering the entire lap/sector if no >0.5G region
        apex_idx = int(np.argmin(v))
        return [
            CornerSegment(
                corner_number=1,
                name="Corner 1",
                start_m=float(s[0]),
                end_m=float(s[-1]),
                nominal_apex_m=float(s[apex_idx]),
            )
        ]

    # Merge adjacent lateral-G islands separated by less than 15m
    merged_bounds: list[tuple[int, int]] = []
    for st, en in zip(starts, ends):
        if merged_bounds and (s[st] - s[merged_bounds[-1][1]]) < 15.0:
            merged_bounds[-1] = (merged_bounds[-1][0], int(en))
        else:
            merged_bounds.append((int(st), int(en)))

    segments: list[CornerSegment] = []
    track_min_s = float(s[0])
    track_max_s = float(s[-1])

    corner_idx = 1
    for st, en in merged_bounds:
        span_m = float(s[en] - s[st])
        if span_m < min_corner_length_m:
            continue
        local_slice = slice(st, en + 1)
        local_min_offset = int(np.argmin(v[local_slice]))
        apex_idx = st + local_min_offset
        apex_s = float(s[apex_idx])

        start_s = max(track_min_s, float(s[st]) - entry_margin_m)
        end_s = min(track_max_s, float(s[en]) + exit_margin_m)

        segments.append(
            CornerSegment(
                corner_number=corner_idx,
                name=f"Turn {corner_idx}",
                start_m=round(start_s, 1),
                end_m=round(end_s, 1),
                nominal_apex_m=round(apex_s, 1),
            )
        )
        corner_idx += 1

    if not segments:
        apex_idx = int(np.argmin(v))
        segments.append(
            CornerSegment(
                corner_number=1,
                name="Turn 1",
                start_m=track_min_s,
                end_m=track_max_s,
                nominal_apex_m=float(s[apex_idx]),
            )
        )

    return segments


def get_track_corners(
    track_slug: str | None,
    distance_m: np.ndarray | None = None,
    lat_accel_g: np.ndarray | None = None,
    speed: np.ndarray | None = None,
) -> list[CornerSegment]:
    """Return pre-mapped corners for a track slug, or auto-segment from telemetry if unmapped."""
    if track_slug:
        normalized = track_slug.strip().lower().replace("_", "-").replace(" ", "-")
        canonical = TRACK_SLUG_ALIASES.get(normalized, normalized)
        if canonical in PREMAPPED_TRACKS:
            corners = PREMAPPED_TRACKS[canonical]
            if distance_m is not None and len(distance_m) > 0:
                max_s = float(np.max(distance_m))
                # Only keep pre-mapped corners that fall within the telemetry distance range
                in_range = [c for c in corners if c.start_m < max_s]
                if in_range:
                    return in_range
            else:
                return corners

    if distance_m is not None and lat_accel_g is not None and speed is not None:
        return auto_segment_corners(distance_m, lat_accel_g, speed)

    return []
