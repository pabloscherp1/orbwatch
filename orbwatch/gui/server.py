"""Local web server for the ORBWATCH tracker interface.

Standard library only. It serves the static interface and a small JSON API:

    GET /api/health
    GET /api/object?norad=25544
    GET /api/track?norad=&t=&before=&after=&step=&lat=&lon=&alt=&site=&mask=
    GET /api/passes?norad=&t=&hours=&lat=&lon=&alt=&site=&mask=

Run it with ``python -m orbwatch.gui`` or ``orbwatch-gui``.

It binds to 127.0.0.1 by default. It is a single-user local tool with no
authentication, and should not be exposed on a network.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from orbwatch.access.passes import GroundSite
from orbwatch.catalog.sources import CatalogFetchError, fetch_celestrak_tle
from orbwatch.catalog.tle import TLE, TleFormatError
from orbwatch.gui import api

STATIC_DIR: Path = Path(__file__).parent / "static"
DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "orbwatch"
DEFAULT_PORT: int = 8765

TleProvider = Callable[[int], TLE]


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


class TrackerRequestHandler(SimpleHTTPRequestHandler):
    """Serves static files from ``STATIC_DIR`` and routes ``/api/``."""

    tle_provider: TleProvider

    def __init__(self, *args: Any, tle_provider: TleProvider, **kwargs: Any) -> None:
        self.tle_provider = tle_provider
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
        raise ApiError(HTTPStatus.NOT_FOUND, f"no such endpoint {path}")

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
) -> ThreadingHTTPServer:
    """Build, but do not start, the tracker server.

    ``tle_provider`` is injectable so tests can serve synthetic element sets
    without touching the network. Port 0 picks a free port.
    """
    if tle_provider is None:
        tle_provider = partial(fetch_celestrak_tle, cache_dir=cache_dir)
    handler = partial(TrackerRequestHandler, tle_provider=tle_provider)
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="orbwatch-gui", description="ORBWATCH live tracker interface"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: binding to {args.host}. This server has no authentication.",
            file=sys.stderr,
        )

    server = create_server(args.host, args.port, cache_dir=args.cache_dir)
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
