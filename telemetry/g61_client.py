"""Garage61 REST API v1 telemetry client.

Base URL: `https://garage61.net/api/v1`

Provides authenticated methods:
- `get_my_latest_lap(car_slug: str = None, track_slug: str = None) -> dict`
- `get_fastest_comparison_lap(car_slug: str, track_slug: str, team_id: str = None) -> dict`
- `download_lap_csv(lap_id: str) -> pd.DataFrame`
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
import pandas as pd
import requests

from config import settings

logger = logging.getLogger(__name__)


class Garage61Error(RuntimeError):
    """Raised when a Garage61 API request fails or returns no matching laps."""


class Garage61Client:
    """Authenticated HTTP client for Garage61 API v1 (`https://garage61.net/api/v1`)."""

    BASE_URL = "https://garage61.net/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = (api_key if api_key is not None else settings.GARAGE61_API_KEY).strip()
        self.base_url = (base_url or settings.GARAGE61_BASE_URL or self.BASE_URL).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        if self.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                }
            )

    def _headers(self, accept: str = "application/json") -> dict[str, str]:
        headers = {"Accept": accept}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_items(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("items", "laps", "data", "results"):
                val = payload.get(key)
                if isinstance(val, list):
                    return val
            if "id" in payload:
                return [payload]
        return []

    def get_my_latest_lap(
        self,
        car_slug: str | None = None,
        track_slug: str | None = None,
    ) -> dict[str, Any]:
        """Resolve the authenticated user's latest lap from `/me/laps`."""
        params: dict[str, Any] = {}
        if car_slug:
            params["car"] = car_slug
        if track_slug:
            params["track"] = track_slug

        url = f"{self.base_url}/me/laps"
        response = self.session.get(url, params=params, headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        laps = self._extract_items(response.json())
        if not laps:
            raise Garage61Error(
                f"No laps found at /me/laps for car={car_slug!r}, track={track_slug!r}."
            )
        return laps[0]

    def get_fastest_comparison_lap(
        self,
        car_slug: str,
        track_slug: str,
        team_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve the fastest comparison lap for the given car and track.

        1. First checks team laps (`/laps?team={team_id}&car={car_slug}&track={track_slug}&sort=lap_time`)
           when `team_id` is provided.
        2. Falls back to the fastest clean public lap (`/laps?car={car_slug}&track={track_slug}&sort=lap_time&clean=true`).
        """
        url = f"{self.base_url}/laps"

        if team_id:
            team_params = {
                "team": team_id,
                "car": car_slug,
                "track": track_slug,
                "sort": "lap_time",
            }
            team_resp = self.session.get(
                url,
                params=team_params,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if team_resp.ok:
                team_laps = self._extract_items(team_resp.json())
                if team_laps:
                    return team_laps[0]

        public_params = {
            "car": car_slug,
            "track": track_slug,
            "sort": "lap_time",
            "clean": "true",
        }
        pub_resp = self.session.get(
            url,
            params=public_params,
            headers=self._headers(),
            timeout=self.timeout,
        )
        pub_resp.raise_for_status()
        public_laps = self._extract_items(pub_resp.json())
        if not public_laps:
            raise Garage61Error(
                f"No comparison laps found for car={car_slug!r}, track={track_slug!r}."
            )
        return public_laps[0]

    def download_lap_csv(self, lap_id: str) -> pd.DataFrame:
        """Download and parse the 60 Hz telemetry CSV stream (`/laps/{lap_id}/csv`)."""
        url = f"{self.base_url}/laps/{lap_id}/csv"
        response = self.session.get(
            url,
            headers=self._headers(accept="text/csv, application/octet-stream, */*"),
            timeout=self.timeout,
        )
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        return normalize_telemetry_columns(df)


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "distance_m": ("distance_m", "LapDist", "Dist", "Distance", "s", "lap_distance"),
    "time_s": ("time_s", "SessionTime", "Time", "LapCurrentLapTime", "t", "time"),
    "speed": ("speed", "Speed", "Velocity", "v", "speed_mps", "speed_kph"),
    "brake": ("brake", "Brake", "BrakeRaw", "brake_pct"),
    "throttle": ("throttle", "Throttle", "ThrottleRaw", "throttle_pct"),
    "steering_deg": ("steering_deg", "SteeringWheelAngle", "Steer", "steering", "delta_deg"),
    "lat_accel_g": ("lat_accel_g", "LatAccel", "ay", "lat_g", "G_lat"),
    "lon_accel_g": ("lon_accel_g", "LongAccel", "ax", "lon_g", "G_lon"),
}


def normalize_telemetry_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Garage61 / iRacing telemetry column names onto canonical names.

    Canonical columns:
    - `distance_m`: meters from start/finish line
    - `time_s`: elapsed lap time in seconds (60 Hz default if absent)
    - `speed`: speed in m/s (or consistent units)
    - `brake`: normalized [0.0, 1.0] brake pedal position
    - `throttle`: normalized [0.0, 1.0] throttle pedal position
    - `steering_deg`: steering wheel angle in degrees
    - `lat_accel_g`: lateral acceleration in G
    - `lon_accel_g`: longitudinal acceleration in G
    """
    out = df.copy()
    existing_lower = {c.lower(): c for c in out.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            match_col = out.columns[out.columns == alias]
            if len(match_col) > 0:
                out[canonical] = out[match_col[0]]
                break
            if alias.lower() in existing_lower:
                out[canonical] = out[existing_lower[alias.lower()]]
                break

    # Convert iRacing SI units when raw iRacing column names are used:
    # iRacing SteeringWheelAngle is in radians; LatAccel/LongAccel are in m/s^2.
    if "SteeringWheelAngle" in df.columns and "steering_deg" not in df.columns:
        out["steering_deg"] = np.degrees(df["SteeringWheelAngle"].astype(float))
    if "LatAccel" in df.columns and "lat_accel_g" not in df.columns:
        lat_vals = df["LatAccel"].astype(float)
        out["lat_accel_g"] = lat_vals / 9.80665 if lat_vals.abs().max() > 6.0 else lat_vals
    if "LongAccel" in df.columns and "lon_accel_g" not in df.columns:
        lon_vals = df["LongAccel"].astype(float)
        out["lon_accel_g"] = lon_vals / 9.80665 if lon_vals.abs().max() > 6.0 else lon_vals

    # Normalize brake and throttle to [0.0, 1.0] if reported in [0, 100]%
    for pedal_col in ("brake", "throttle"):
        if pedal_col in out.columns:
            vals = out[pedal_col].astype(float)
            if vals.max() > 1.5:
                out[pedal_col] = vals / 100.0

    if "time_s" not in out.columns and len(out) > 0:
        out["time_s"] = np.arange(len(out), dtype=float) / 60.0

    return out


# Module-level convenience functions matching the specification signature
def get_my_latest_lap(car_slug: str | None = None, track_slug: str | None = None) -> dict[str, Any]:
    """Resolve the user's latest lap from `/me/laps`."""
    return Garage61Client().get_my_latest_lap(car_slug=car_slug, track_slug=track_slug)


def get_fastest_comparison_lap(
    car_slug: str,
    track_slug: str,
    team_id: str | None = None,
) -> dict[str, Any]:
    """Resolve the fastest comparison lap (team first, then fastest clean public lap)."""
    return Garage61Client().get_fastest_comparison_lap(
        car_slug=car_slug,
        track_slug=track_slug,
        team_id=team_id,
    )


def download_lap_csv(lap_id: str) -> pd.DataFrame:
    """Download and parse the 60 Hz CSV telemetry stream (`/laps/{lap_id}/csv`)."""
    return Garage61Client().download_lap_csv(lap_id=lap_id)
