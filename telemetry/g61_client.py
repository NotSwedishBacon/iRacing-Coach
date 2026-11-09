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
    def _csv_param(value: list[Any] | tuple[Any, ...] | str | int | None) -> str | None:
        """Serialize a list or scalar parameter into a comma-separated string (`explode=false`)."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)

    @staticmethod
    def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
        """Filter out None values from query parameters."""
        return {k: v for k, v in params.items() if v is not None}

    def _request_json(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Execute an HTTP request against Garage 61 API v1 and return parsed JSON or status."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        clean_q = self._clean_params(params) if params else None
        method_upper = method.upper()

        kwargs: dict[str, Any] = {
            "headers": self._headers(),
            "timeout": self.timeout,
        }
        if clean_q:
            kwargs["params"] = clean_q
        if json_body is not None:
            kwargs["json"] = json_body

        http_fn = getattr(self.session, method_upper.lower(), None)
        if http_fn is not None and method_upper in ("GET", "POST", "DELETE", "PUT", "PATCH"):
            response = http_fn(url, **kwargs)
        else:
            response = self.session.request(method_upper, url, **kwargs)

        response.raise_for_status()
        if response.status_code in (201, 204) and not getattr(response, "text", "").strip():
            return {"status": "success", "status_code": response.status_code}
        try:
            return response.json()
        except Exception:
            return {"status": "success", "status_code": getattr(response, "status_code", 200)}

    def _request_bytes(self, path: str, accept: str = "application/octet-stream, */*") -> bytes:
        """Execute a GET request returning raw binary bytes."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(
            url,
            headers=self._headers(accept=accept),
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = getattr(response, "content", None)
        if isinstance(content, bytes):
            return content
        text = getattr(response, "text", "")
        return text.encode("utf-8") if isinstance(text, str) else b""

    def _request_text(self, path: str, accept: str = "text/csv, application/octet-stream, */*") -> str:
        """Execute a GET request returning raw text (e.g., CSV)."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(
            url,
            headers=self._headers(accept=accept),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

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

    # -------------------------------------------------------------------------
    # High-level coaching convenience methods
    # -------------------------------------------------------------------------

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
        csv_text = self.get_lap_csv(lap_id)
        df = pd.read_csv(io.StringIO(csv_text))
        return normalize_telemetry_columns(df)

    # -------------------------------------------------------------------------
    # Complete Garage 61 OpenAPI v1 Endpoints (all 43 operations)
    # -------------------------------------------------------------------------

    # 1. GET /analyses (operationId: findAnalyses)
    def find_analyses(
        self,
        types: list[str] | str | None = None,
        trainingplan: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Analyses for current user (`GET /analyses`)."""
        return self._request_json(
            "GET",
            "/analyses",
            params={
                "types": self._csv_param(types),
                "trainingplan": trainingplan,
                "limit": limit,
                "offset": offset,
            },
        )

    # 2. GET /analyses/{id} (operationId: getAnalysis)
    def get_analysis(self, id: str) -> dict[str, Any] | list[Any]:
        """Get analysis details (`GET /analyses/{id}`)."""
        return self._request_json("GET", f"/analyses/{id}")

    # 3. GET /car-groups (operationId: findCarGroups)
    def find_car_groups(self) -> dict[str, Any] | list[Any]:
        """Available car groups (`GET /car-groups`)."""
        return self._request_json("GET", "/car-groups")

    # 4. GET /cars (operationId: findCars)
    def find_cars(self) -> dict[str, Any] | list[Any]:
        """Available cars (`GET /cars`)."""
        return self._request_json("GET", "/cars")

    # 5. GET /events (operationId: findEvents)
    def find_events(
        self,
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
        """Find events (`GET /events`)."""
        return self._request_json(
            "GET",
            "/events",
            params={
                "platform": platform,
                "platformEvent": platform_event,
                "after": after,
                "before": before,
                "car": car,
                "track": track,
                "user": user,
                "team": team,
                "limit": limit,
                "offset": offset,
            },
        )

    # 6. GET /events/{id} (operationId: getEvent)
    def get_event(self, id: str) -> dict[str, Any] | list[Any]:
        """Get event details (`GET /events/{id}`)."""
        return self._request_json("GET", f"/events/{id}")

    # 7. GET /laps (operationId: findLaps)
    def find_laps(
        self,
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
        """Find laps and lap records (`GET /laps`)."""
        return self._request_json(
            "GET",
            "/laps",
            params={
                "cars": self._csv_param(cars),
                "tracks": self._csv_param(tracks),
                "seasons": self._csv_param(seasons),
                "drivers": self._csv_param(drivers),
                "teams": self._csv_param(teams),
                "extraDrivers": self._csv_param(extra_drivers),
                "age": age,
                "after": after,
                "sessionTypes": self._csv_param(session_types),
                "sessionSetupTypes": self._csv_param(session_setup_types),
                "seeTelemetry": see_telemetry,
                "seeGhostLap": see_ghost_lap,
                "seeSetup": see_setup,
                "lapTypes": self._csv_param(lap_types),
                "pushToPass": push_to_pass,
                "event": event,
                "unclean": unclean,
                "minWeightPenalty": min_weight_penalty,
                "maxWeightPenalty": max_weight_penalty,
                "minPowerAdjust": min_power_adjust,
                "maxPowerAdjust": max_power_adjust,
                "minRating": min_rating,
                "maxRating": max_rating,
                "minFuel": min_fuel,
                "maxFuel": max_fuel,
                "minFuelUsed": min_fuel_used,
                "maxFuelUsed": max_fuel_used,
                "minLapTime": min_lap_time,
                "maxLapTime": max_lap_time,
                "minConditionsTrackUsage": min_conditions_track_usage,
                "maxConditionsTrackUsage": max_conditions_track_usage,
                "minConditionsTrackWetness": min_conditions_track_wetness,
                "maxConditionsTrackWetness": max_conditions_track_wetness,
                "minConditionsTrackTemp": min_conditions_track_temp,
                "maxConditionsTrackTemp": max_conditions_track_temp,
                "minConditionsAirTemp": min_conditions_air_temp,
                "maxConditionsAirTemp": max_conditions_air_temp,
                "minConditionsWindVel": min_conditions_wind_vel,
                "maxConditionsWindVel": max_conditions_wind_vel,
                "minConditionsRelativeHumidity": min_conditions_relative_humidity,
                "maxConditionsRelativeHumidity": max_conditions_relative_humidity,
                "minConditionsFogLevel": min_conditions_fog_level,
                "maxConditionsFogLevel": max_conditions_fog_level,
                "minConditionsPrecipitation": min_conditions_precipitation,
                "maxConditionsPrecipitation": max_conditions_precipitation,
                "minConditionsCloud": min_conditions_cloud,
                "maxConditionsCloud": max_conditions_cloud,
                "conditionsWindDir": self._csv_param(conditions_wind_dir),
                "round": round_units,
                "group": group,
                "limit": limit,
                "offset": offset,
            },
        )

    # 8. GET /laps/{id} (operationId: getLap)
    def get_lap(self, id: str) -> dict[str, Any] | list[Any]:
        """Get information about a specific lap (`GET /laps/{id}`)."""
        return self._request_json("GET", f"/laps/{id}")

    # 9. GET /laps/{id}/csv (operationId: getLapCSV)
    def get_lap_csv(self, id: str) -> str:
        """Export the telemetry for a lap as a CSV file (`GET /laps/{id}/csv`)."""
        return self._request_text(f"/laps/{id}/csv", accept="text/csv, application/octet-stream, */*")

    # 10. GET /laps/{id}/ghost.bin (operationId: getLapGhost)
    def get_lap_ghost(self, id: str) -> bytes:
        """Download the ghost lap file for a lap (`GET /laps/{id}/ghost.bin`)."""
        return self._request_bytes(f"/laps/{id}/ghost.bin")

    # 11. GET /me (operationId: me)
    def get_me(self) -> dict[str, Any] | list[Any]:
        """Information about the currently authenticated user (`GET /me`)."""
        return self._request_json("GET", "/me")

    # 12. GET /me/accounts (operationId: getAccounts)
    def get_accounts(self, rating_history: bool | None = None) -> dict[str, Any] | list[Any]:
        """Get linked accounts for the current user (`GET /me/accounts`)."""
        return self._request_json(
            "GET",
            "/me/accounts",
            params={"ratingHistory": rating_history},
        )

    # 13. POST /me/datapackgroups (operationId: createUserDataPackGroup)
    def create_user_data_pack_group(self, id: str) -> dict[str, Any] | list[Any]:
        """Subscribe user to a data pack group (`POST /me/datapackgroups`)."""
        return self._request_json("POST", "/me/datapackgroups", json_body={"id": id})

    # 14. DELETE /me/datapackgroups/{id} (operationId: deleteUserDataPackGroup)
    def delete_user_data_pack_group(self, id: str) -> dict[str, Any] | list[Any]:
        """Unsubscribe user from a data pack group (`DELETE /me/datapackgroups/{id}`)."""
        return self._request_json("DELETE", f"/me/datapackgroups/{id}")

    # 15. POST /me/datapacks (operationId: createUserDataPack)
    def create_user_data_pack(self, id: str) -> dict[str, Any] | list[Any]:
        """Subscribe user to a data pack (`POST /me/datapacks`)."""
        return self._request_json("POST", "/me/datapacks", json_body={"id": id})

    # 16. DELETE /me/datapacks/{id} (operationId: deleteUserDataPack)
    def delete_user_data_pack(self, id: str) -> dict[str, Any] | list[Any]:
        """Unsubscribe user from a data pack (`DELETE /me/datapacks/{id}`)."""
        return self._request_json("DELETE", f"/me/datapacks/{id}")

    # 17. GET /me/statistics (operationId: getStatistics)
    def get_statistics(
        self,
        start: str | None = None,
        end: str | None = None,
        cars: list[int] | str | int | None = None,
        tracks: list[int] | str | int | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Get personal driving statistics (`GET /me/statistics`)."""
        return self._request_json(
            "GET",
            "/me/statistics",
            params={
                "start": start,
                "end": end,
                "cars": self._csv_param(cars),
                "tracks": self._csv_param(tracks),
            },
        )

    # 18. GET /platforms (operationId: findPlatforms)
    def find_platforms(self) -> dict[str, Any] | list[Any]:
        """Available platforms (`GET /platforms`)."""
        return self._request_json("GET", "/platforms")

    # 19. GET /teams (operationId: findTeams)
    def find_teams(self) -> dict[str, Any] | list[Any]:
        """Joined teams (`GET /teams`)."""
        return self._request_json("GET", "/teams")

    # 20. GET /teams/{id} (operationId: getTeam)
    def get_team(self, id: str) -> dict[str, Any] | list[Any]:
        """Get information about a specific team (`GET /teams/{id}`)."""
        return self._request_json("GET", f"/teams/{id}")

    # 21. GET /teams/{team}/datapackgroups (operationId: getTeamDataPackGroups)
    def get_team_data_pack_groups(
        self,
        team: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Get data pack groups for a team (`GET /teams/{team}/datapackgroups`)."""
        return self._request_json(
            "GET",
            f"/teams/{team}/datapackgroups",
            params={"limit": limit, "offset": offset},
        )

    # 22. GET /teams/{team}/datapackgroups/{id} (operationId: getTeamDataPackGroup)
    def get_team_data_pack_group(self, team: str, id: str) -> dict[str, Any] | list[Any]:
        """Get data pack group for a team (`GET /teams/{team}/datapackgroups/{id}`)."""
        return self._request_json("GET", f"/teams/{team}/datapackgroups/{id}")

    # 23. POST /teams/{team}/datapackgroups/{id}/members (operationId: createDataPackGroupMember)
    def create_data_pack_group_member(
        self,
        team: str,
        id: str,
        data_pack_id: str,
    ) -> dict[str, Any] | list[Any]:
        """Add data pack to a data pack group (`POST /teams/{team}/datapackgroups/{id}/members`)."""
        return self._request_json(
            "POST",
            f"/teams/{team}/datapackgroups/{id}/members",
            json_body={"id": data_pack_id},
        )

    # 24. DELETE /teams/{team}/datapackgroups/{id}/members/{member} (operationId: deleteDataPackGroupMember)
    def delete_data_pack_group_member(
        self,
        team: str,
        id: str,
        member: str,
    ) -> dict[str, Any] | list[Any]:
        """Remove data pack from a data pack group (`DELETE /teams/{team}/datapackgroups/{id}/members/{member}`)."""
        return self._request_json("DELETE", f"/teams/{team}/datapackgroups/{id}/members/{member}")

    # 25. DELETE /teams/{team}/datapackgroups/{id}/subscribers/{subscriber} (operationId: deleteDataPackGroupSubscriber)
    def delete_data_pack_group_subscriber(
        self,
        team: str,
        id: str,
        subscriber: str,
    ) -> dict[str, Any] | list[Any]:
        """Remove subscriber from a data pack group (`DELETE /teams/{team}/datapackgroups/{id}/subscribers/{subscriber}`)."""
        return self._request_json(
            "DELETE",
            f"/teams/{team}/datapackgroups/{id}/subscribers/{subscriber}",
        )

    # 26. GET /teams/{team}/datapacks (operationId: getTeamDataPacks)
    def get_team_data_packs(
        self,
        team: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Get data packs for a team (`GET /teams/{team}/datapacks`)."""
        return self._request_json(
            "GET",
            f"/teams/{team}/datapacks",
            params={"limit": limit, "offset": offset},
        )

    # 27. GET /teams/{team}/datapacks/{id} (operationId: getTeamDataPack)
    def get_team_data_pack(self, team: str, id: str) -> dict[str, Any] | list[Any]:
        """Get data pack for a team (`GET /teams/{team}/datapacks/{id}`)."""
        return self._request_json("GET", f"/teams/{team}/datapacks/{id}")

    # 28. POST /teams/{team}/datapacks/{id} (operationId: updateTeamDataPack)
    def update_team_data_pack(
        self,
        team: str,
        id: str,
        name: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Update data pack for a team (`POST /teams/{team}/datapacks/{id}`)."""
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if archived is not None:
            body["archived"] = archived
        return self._request_json("POST", f"/teams/{team}/datapacks/{id}", json_body=body)

    # 29. POST /teams/{team}/datapacks/{id}/content (operationId: createTeamDataPackContent)
    def create_team_data_pack_content(
        self,
        team: str,
        id: str,
        video_url: str,
        content_type: str = "video",
        featured: bool = True,
    ) -> dict[str, Any] | list[Any]:
        """Add content to a data pack for a team (`POST /teams/{team}/datapacks/{id}/content`)."""
        return self._request_json(
            "POST",
            f"/teams/{team}/datapacks/{id}/content",
            json_body={
                "type": content_type,
                "videoUrl": video_url,
                "featured": featured,
            },
        )

    # 30. POST /teams/{team}/datapacks/{id}/content/{item} (operationId: updateTeamDataPackContent)
    def update_team_data_pack_content(
        self,
        team: str,
        id: str,
        item: str,
        video_url: str,
        content_type: str = "video",
    ) -> dict[str, Any] | list[Any]:
        """Update data pack content for a team (`POST /teams/{team}/datapacks/{id}/content/{item}`)."""
        return self._request_json(
            "POST",
            f"/teams/{team}/datapacks/{id}/content/{item}",
            json_body={
                "type": content_type,
                "videoUrl": video_url,
            },
        )

    # 31. DELETE /teams/{team}/datapacks/{id}/content/{item} (operationId: deleteTeamDataPackContent)
    def delete_team_data_pack_content(
        self,
        team: str,
        id: str,
        item: str,
    ) -> dict[str, Any] | list[Any]:
        """Delete data pack content for a team (`DELETE /teams/{team}/datapacks/{id}/content/{item}`)."""
        return self._request_json("DELETE", f"/teams/{team}/datapacks/{id}/content/{item}")

    # 32. GET /teams/{team}/datapacks/{id}/content/{item}/ghost.bin (operationId: getTeamDataPackContentGhostLapBin)
    def get_team_data_pack_content_ghost_lap_bin(
        self,
        team: str,
        id: str,
        item: str,
    ) -> bytes:
        """Download ghost lap file for a content item (`GET /teams/{team}/datapacks/{id}/content/{item}/ghost.bin`)."""
        return self._request_bytes(f"/teams/{team}/datapacks/{id}/content/{item}/ghost.bin")

    # 33. GET /teams/{team}/datapacks/{id}/content/{item}/lap.csv (operationId: getTeamDataPackContentLapCsv)
    def get_team_data_pack_content_lap_csv(
        self,
        team: str,
        id: str,
        item: str,
    ) -> str:
        """Export telemetry for a lap content item (`GET /teams/{team}/datapacks/{id}/content/{item}/lap.csv`)."""
        return self._request_text(
            f"/teams/{team}/datapacks/{id}/content/{item}/lap.csv",
            accept="text/csv, application/octet-stream, */*",
        )

    # 34. POST /teams/{team}/datapacks/{id}/content/{item}/move (operationId: moveTeamDataPackContent)
    def move_team_data_pack_content(
        self,
        team: str,
        id: str,
        item: str,
        position: int,
    ) -> dict[str, Any] | list[Any]:
        """Move data pack content for a team (`POST /teams/{team}/datapacks/{id}/content/{item}/move`)."""
        return self._request_json(
            "POST",
            f"/teams/{team}/datapacks/{id}/content/{item}/move",
            json_body={"position": position},
        )

    # 35. GET /teams/{team}/datapacks/{id}/content/{item}/replay.bin (operationId: getTeamDataPackContentReplayBin)
    def get_team_data_pack_content_replay_bin(
        self,
        team: str,
        id: str,
        item: str,
    ) -> bytes:
        """Download file for a replay content item (`GET /teams/{team}/datapacks/{id}/content/{item}/replay.bin`)."""
        return self._request_bytes(f"/teams/{team}/datapacks/{id}/content/{item}/replay.bin")

    # 36. GET /teams/{team}/datapacks/{id}/content/{item}/setup.sto (operationId: getTeamDataPackContentSetupSto)
    def get_team_data_pack_content_setup_sto(
        self,
        team: str,
        id: str,
        item: str,
    ) -> bytes:
        """Download iRacing setup for a setup content item (`GET /teams/{team}/datapacks/{id}/content/{item}/setup.sto`)."""
        return self._request_bytes(f"/teams/{team}/datapacks/{id}/content/{item}/setup.sto")

    # 37. DELETE /teams/{team}/datapacks/{id}/subscribers/{subscriber} (operationId: deleteDataPackSubscriber)
    def delete_data_pack_subscriber(
        self,
        team: str,
        id: str,
        subscriber: str,
    ) -> dict[str, Any] | list[Any]:
        """Remove subscriber from a data pack (`DELETE /teams/{team}/datapacks/{id}/subscribers/{subscriber}`)."""
        return self._request_json("DELETE", f"/teams/{team}/datapacks/{id}/subscribers/{subscriber}")

    # 38. POST /teams/{team}/invites (operationId: createTeamInvite)
    def create_team_invite(
        self,
        team: str,
        name: str,
        max_usages: int = 0,
        expires_after: int = 0,
    ) -> dict[str, Any] | list[Any]:
        """Create team invite (`POST /teams/{team}/invites`)."""
        return self._request_json(
            "POST",
            f"/teams/{team}/invites",
            json_body={
                "name": name,
                "maxUsages": max_usages,
                "expiresAfter": expires_after,
            },
        )

    # 39. DELETE /teams/{team}/members/{id} (operationId: deleteTeamMember)
    def delete_team_member(self, team: str, id: str) -> dict[str, Any] | list[Any]:
        """Remove member from team (`DELETE /teams/{team}/members/{id}`)."""
        return self._request_json("DELETE", f"/teams/{team}/members/{id}")

    # 40. GET /teams/{team}/statistics (operationId: getTeamStatistics)
    def get_team_statistics(
        self,
        team: str,
        start: str | None = None,
        end: str | None = None,
        cars: list[int] | str | int | None = None,
        tracks: list[int] | str | int | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Get team driving statistics (`GET /teams/{team}/statistics`)."""
        return self._request_json(
            "GET",
            f"/teams/{team}/statistics",
            params={
                "start": start,
                "end": end,
                "cars": self._csv_param(cars),
                "tracks": self._csv_param(tracks),
            },
        )

    # 41. GET /teams/{team}/trainingplans (operationId: getTeamTrainingPlans)
    def get_team_training_plans(
        self,
        team: str,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any] | list[Any]:
        """Get training plans for a team (`GET /teams/{team}/trainingplans`)."""
        return self._request_json(
            "GET",
            f"/teams/{team}/trainingplans",
            params={"limit": limit, "offset": offset},
        )

    # 42. GET /teams/{team}/trainingplans/{id} (operationId: getTeamTrainingPlan)
    def get_team_training_plan(self, team: str, id: str) -> dict[str, Any] | list[Any]:
        """Get training plan for a team (`GET /teams/{team}/trainingplans/{id}`)."""
        return self._request_json("GET", f"/teams/{team}/trainingplans/{id}")

    # 43. GET /tracks (operationId: findTracks)
    def find_tracks(self) -> dict[str, Any] | list[Any]:
        """Available tracks (`GET /tracks`)."""
        return self._request_json("GET", "/tracks")


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

