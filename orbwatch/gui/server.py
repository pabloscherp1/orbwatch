"""Local web server for the ORBWATCH tracker interface.

Standard library only. It serves the static interface and a small JSON API:

    GET /api/health
    GET /api/object?norad=25544
    GET /api/track?norad=&t=&before=&after=&step=&lat=&lon=&alt=&site=&mask=
    GET /api/passes?norad=&t=&hours=&lat=&lon=&alt=&site=&mask=
    GET /api/behaviour?norad=&days=&sigmas=
    GET /api/mission?norad=&altitude=&offset=&dry=&isp=&inj_alt=&inj_inc=&prox=&ops=
        &disposal=

Run it with ``python -m orbwatch.gui`` or ``orbwatch-gui``.

The behaviour endpoint needs Space-Track credentials, read from ``.env`` in the
directory the server is started from, as for ``python -m orbwatch.events``.
Everything else works without them.

It binds to 127.0.0.1 by default. It is a single-user local tool with no
authentication, and should not be exposed on a network.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from collections import OrderedDict
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from orbwatch.access.passes import GroundSite
from orbwatch.catalog.sources import CatalogFetchError, fetch_celestrak_tle
from orbwatch.catalog.spacetrack import (
    DEFAULT_CACHE_DIR as SPACETRACK_CACHE_DIR,
)
from orbwatch.catalog.spacetrack import (
    AuthenticationError,
    SpaceTrackClient,
    SpaceTrackError,
)
from orbwatch.catalog.tle import TLE, TleFormatError
from orbwatch.gui import api
from orbwatch.transfer.mission import MissionAssumptions

STATIC_DIR: Path = Path(__file__).parent / "static"
DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "orbwatch"
DEFAULT_PORT: int = 8765
DEFAULT_HISTORY_DAYS: int = 365
HISTORY_DAYS_RANGE: tuple[int, int] = (14, 1095)
SIGMAS_RANGE: tuple[float, float] = (3.0, 10.0)
BEHAVIOUR_CACHE_SIZE: int = 16
MISSION_CACHE_SIZE: int = 32

MISSION_PARAMETERS: dict[str, tuple[str, float, float, float]] = {
    "altitude": ("dropoff_altitude_km", 525.0, 300.0, 1200.0),
    "offset": ("ltan_offset_h", -1.0, -6.0, 6.0),
    "dry": ("dry_mass_kg", 150.0, 1.0, 10000.0),
    "isp": ("isp_s", 220.0, 30.0, 500.0),
    "inj_alt": ("injection_altitude_error_km", 10.0, 0.0, 100.0),
    "inj_inc": ("injection_inclination_error_deg", 0.1, 0.0, 2.0),
    "prox": ("proximity_allocation_m_s", 10.0, 0.0, 500.0),
    "ops": ("operations_days", 90.0, 0.0, 3650.0),
    "disposal": ("disposal_perigee_km", 300.0, 100.0, 1000.0),
}
"""Query name: (argument, default, minimum, maximum)."""

TleProvider = Callable[[int], TLE]
HistoryProvider = Callable[[int, date, date], Sequence[TLE]]


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _single(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name)
    return values[0] if values else None


def _require(query: dict[str, list[str]], name: str) -> str:
    value = _single(query, name)
    if value is None or value == "":
        raise ApiError(HTTPStatus.BAD_REQUEST, f"missing query parameter '{name}'")
    return value


def _float(
    query: dict[str, list[str]], name: str, default: float | None = None
) -> float:
    raw = _single(query, name)
    if raw is None or raw == "":
        if default is None:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"missing query parameter '{name}'")
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' must be a number") from error
    if value != value or value in (float("inf"), float("-inf")):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"'{name}' must be finite")
    return value


def _norad(query: dict[str, list[str]]) -> int:
    raw = _require(query, "norad")
    if not raw.isdigit() or not 0 < int(raw) < 1_000_000_000:
        raise ApiError(HTTPStatus.BAD_REQUEST, "'norad' must be a positive integer")
    return int(raw)


def _time(query: dict[str, list[str]]) -> datetime:
    raw = _single(query, "t")
    if raw is None or raw == "":
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as error:
        raise ApiError(
            HTTPStatus.BAD_REQUEST, "'t' must be an ISO 8601 time"
        ) from error
    if parsed.tzinfo is None:
        raise ApiError(HTTPStatus.BAD_REQUEST, "'t' must include a timezone, e.g. Z")
    return parsed.astimezone(UTC)


def _site(query: dict[str, list[str]]) -> GroundSite:
    name = (_single(query, "site") or "Site")[:60]
    try:
        return GroundSite(
            name=name,
            lat_deg=_float(query, "lat"),
            lon_deg=_float(query, "lon"),
            alt_km=_float(query, "alt", 0.0),
        )
    except ValueError as error:
        raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from error


class SpaceTrackHistory:
    """Element set histories from Space-Track, one request at a time.

    The client is created on first use, so the tracker starts and runs without
    credentials and only the behaviour view asks for them. The lock keeps
    concurrent browser requests from overrunning Space-Track's rate limits.
    """

    def __init__(self, env_file: Path, cache_dir: Path) -> None:
        self.env_file = env_file
        self.cache_dir = cache_dir
        self._client: SpaceTrackClient | None = None
        self._lock = threading.Lock()

    def __call__(self, norad_id: int, start: date, stop: date) -> Sequence[TLE]:
        with self._lock:
            if self._client is None:
                self._client = SpaceTrackClient.from_env(self.env_file, self.cache_dir)
            return self._client.element_set_history(norad_id, start, stop)


class BehaviourService:
    """Runs the events analysis for the interface and remembers recent results.

    Keyed by the UTC date, so a history is re-analysed once a day at most.
    """

    def __init__(self, history_provider: HistoryProvider) -> None:
        self._history = history_provider
        self._cache: OrderedDict[tuple[int, int, float, date], dict[str, Any]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def payload(
        self, norad_id: int, days: int, sigmas: float, today: date
    ) -> dict[str, Any]:
        key = (norad_id, days, sigmas, today)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            records = self._history(norad_id, today - timedelta(days=days), today)
            if not records:
                raise ApiError(
                    HTTPStatus.NOT_FOUND,
                    f"Space-Track has no element sets for NORAD {norad_id} "
                    f"in the last {days} days",
                )
            try:
                result = api.behaviour(records, sigmas=sigmas)
            except ValueError as error:
                raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error
            self._cache[key] = result
            while len(self._cache) > BEHAVIOUR_CACHE_SIZE:
                self._cache.popitem(last=False)
            return result


class TrackerRequestHandler(SimpleHTTPRequestHandler):
    """Serves static files from ``STATIC_DIR`` and routes ``/api/``."""

    tle_provider: TleProvider
    behaviour: BehaviourService
    mission_cache: OrderedDict[Any, dict[str, Any]]
    mission_cache_lock: threading.Lock

    def __init__(
        self,
        *args: Any,
        tle_provider: TleProvider,
        behaviour: BehaviourService,
        mission_cache: OrderedDict[Any, dict[str, Any]],
        mission_cache_lock: threading.Lock,
        **kwargs: Any,
    ) -> None:
        self.tle_provider = tle_provider
        self.behaviour = behaviour
        self.mission_cache = mission_cache
        self.mission_cache_lock = mission_cache_lock
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    # -- static ------------------------------------------------------------

    def list_directory(self, path: str) -> None:  # type: ignore[override]
        self.send_error(HTTPStatus.NOT_FOUND, "directory listing disabled")
        return None

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if isinstance(code, HTTPStatus):
            code = code.value
        if str(code).isdigit() and int(code) >= 400:
            super().log_request(code, size)

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        if not parts.path.startswith("/api/"):
            super().do_GET()
            return

        query = parse_qs(parts.query)
        try:
            payload = self._route(parts.path, query)
            self._send_json(HTTPStatus.OK, payload)
        except ApiError as error:
            self._send_json(error.status, {"error": error.message})
        except Exception as error:  # noqa: BLE001, the server must not die on one request
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(error).__name__}: {error}"},
            )

    def _route(self, path: str, query: dict[str, list[str]]) -> dict[str, Any]:
        if path == "/api/health":
            return {"status": "ok", "time_utc": datetime.now(UTC).isoformat()}
        if path == "/api/object":
            return api.object_summary(self._tle(_norad(query)), datetime.now(UTC))
        if path == "/api/track":
            tle = self._tle(_norad(query))
            try:
                return api.track(
                    tle,
                    _site(query),
                    _time(query),
                    before_s=_float(query, "before", 3600.0),
                    after_s=_float(query, "after", 3600.0),
                    step_s=_float(query, "step", 20.0),
                    min_elevation_deg=_float(query, "mask", 0.0),
                )
            except ValueError as error:
                raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from error
        if path == "/api/passes":
            tle = self._tle(_norad(query))
            try:
                return api.passes(
                    tle,
                    _site(query),
                    _time(query),
                    hours=_float(query, "hours", 24.0),
                    min_elevation_deg=_float(query, "mask", 0.0),
                )
            except ValueError as error:
                raise ApiError(HTTPStatus.BAD_REQUEST, str(error)) from error
        if path == "/api/behaviour":
            return self._behaviour(query)
        if path == "/api/mission":
            return self._mission(query)
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such endpoint {path}")

    def _mission(self, query: dict[str, list[str]]) -> dict[str, Any]:
        norad_id = _norad(query)
        values: dict[str, float] = {}
        for name, (argument, default, low, high) in MISSION_PARAMETERS.items():
            value = _float(query, name, default)
            if not low <= value <= high:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST, f"'{name}' must be in [{low:g}, {high:g}]"
                )
            values[argument] = value
        epoch = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        key = (norad_id, epoch, tuple(sorted(values.items())))
        with self.mission_cache_lock:
            if key in self.mission_cache:
                self.mission_cache.move_to_end(key)
                return self.mission_cache[key]
        tle = self._tle(norad_id)
        assumptions = MissionAssumptions(
            **{
                k: v
                for k, v in values.items()
                if k in MissionAssumptions.__dataclass_fields__
            }
        )
        try:
            result = api.mission(
                tle,
                epoch,
                dropoff_altitude_km=values["dropoff_altitude_km"],
                ltan_offset_h=values["ltan_offset_h"],
                dry_mass_kg=values["dry_mass_kg"],
                isp_s=values["isp_s"],
                assumptions=assumptions,
            )
        except ValueError as error:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, str(error)) from error
        with self.mission_cache_lock:
            self.mission_cache[key] = result
            while len(self.mission_cache) > MISSION_CACHE_SIZE:
                self.mission_cache.popitem(last=False)
        return result

    def _behaviour(self, query: dict[str, list[str]]) -> dict[str, Any]:
        norad_id = _norad(query)
        days = _float(query, "days", float(DEFAULT_HISTORY_DAYS))
        lo, hi = HISTORY_DAYS_RANGE
        if days != int(days) or not lo <= days <= hi:
            raise ApiError(
                HTTPStatus.BAD_REQUEST, f"'days' must be a whole number in [{lo}, {hi}]"
            )
        sigmas = _float(query, "sigmas", api.DEFAULT_SIGMAS)
        if not SIGMAS_RANGE[0] <= sigmas <= SIGMAS_RANGE[1]:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"'sigmas' must be in [{SIGMAS_RANGE[0]:g}, {SIGMAS_RANGE[1]:g}]",
            )
        try:
            return self.behaviour.payload(
                norad_id, int(days), sigmas, datetime.now(UTC).date()
            )
        except AuthenticationError as error:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, str(error)) from error
        except SpaceTrackError as error:
            raise ApiError(HTTPStatus.BAD_GATEWAY, str(error)) from error

    def _tle(self, norad_id: int) -> TLE:
        try:
            return self.tle_provider(norad_id)
        except CatalogFetchError as error:
            status = (
                HTTPStatus.NOT_FOUND
                if "no current element set" in str(error)
                else HTTPStatus.BAD_GATEWAY
            )
            raise ApiError(status, str(error)) from error
        except TleFormatError as error:
            raise ApiError(
                HTTPStatus.BAD_GATEWAY, f"malformed element set: {error}"
            ) from error

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def create_server(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    tle_provider: TleProvider | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    history_provider: HistoryProvider | None = None,
    env_file: Path = Path(".env"),
    history_cache_dir: Path = SPACETRACK_CACHE_DIR,
) -> ThreadingHTTPServer:
    """Build, but do not start, the tracker server.

    ``tle_provider`` and ``history_provider`` are injectable so tests can serve
    synthetic element sets without touching the network. Port 0 picks a free
    port.
    """
    if tle_provider is None:
        tle_provider = partial(fetch_celestrak_tle, cache_dir=cache_dir)
    if history_provider is None:
        history_provider = SpaceTrackHistory(
            env_file.resolve(), history_cache_dir.resolve()
        )
    handler = partial(
        TrackerRequestHandler,
        tle_provider=tle_provider,
        behaviour=BehaviourService(history_provider),
        mission_cache=OrderedDict(),
        mission_cache_lock=threading.Lock(),
    )
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="orbwatch-gui", description="ORBWATCH live tracker interface"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Space-Track credentials for the behaviour view (default ./.env)",
    )
    parser.add_argument(
        "--history-cache-dir",
        type=Path,
        default=SPACETRACK_CACHE_DIR,
        help="where downloaded histories are kept (default ./data/cache/spacetrack)",
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: binding to {args.host}. This server has no authentication.",
            file=sys.stderr,
        )

    server = create_server(
        args.host,
        args.port,
        cache_dir=args.cache_dir,
        env_file=args.env_file,
        history_cache_dir=args.history_cache_dir,
    )
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"ORBWATCH tracker running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0
