"""Historical element sets from Space-Track.

Celestrak serves only an object's *current* element set. Anything that asks what
an object has been doing, manoeuvre detection above all, needs the archive, and
the archive lives behind an account at https://www.space-track.org.

**Credentials** come from ``SPACETRACK_USER`` and ``SPACETRACK_PASSWORD``, in the
environment or in a ``.env`` file that is git-ignored. They are never written to
disk by this module, never logged, and never included in an exception message or
a repr.

**Politeness matters here.** Space-Track's documented limits, read on
2026-09-20, are fewer than 30 requests per minute and 300 per hour, and for the
historical ``gp_history`` class the guidance is stronger: download once and store
it locally rather than querying repeatedly. So this client throttles itself and
caches every response permanently by default. A historical query is immutable:
the element sets published for a past date range never change.

**Authentication is judged on the body, not the status.** A wrong password gets
HTTP 200 with ``{"Login":"Failed"}``, verified against the live service on
2026-09-20. Trusting the status code would leave an unauthenticated session that
fails later with a confusing 401.
"""

from __future__ import annotations

import hashlib
import http.cookiejar
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from orbwatch.catalog.sources import USER_AGENT, CatalogFetchError, ssl_context
from orbwatch.catalog.tle import TLE, parse_tle_text

SPACETRACK_BASE: str = "https://www.space-track.org"
LOGIN_URL: str = f"{SPACETRACK_BASE}/ajaxauth/login"
QUERY_URL: str = f"{SPACETRACK_BASE}/basicspacedata/query"

DEFAULT_CACHE_DIR: Path = Path("data/cache/spacetrack")

MIN_REQUEST_INTERVAL_S: float = 3.0
"""Self-imposed gap between requests, giving at most 20 per minute."""

TLE_FORMAT_MAX_NORAD_ID: int = 100000
"""The legacy TLE output cannot express catalogue numbers at or above this."""

_HTTP_TIMEOUT_S: float = 120.0


class SpaceTrackError(CatalogFetchError):
    """Raised when Space-Track cannot be reached or refuses a request."""


class AuthenticationError(SpaceTrackError):
    """Raised when Space-Track rejects the credentials."""


def load_dotenv(path: Path | str = ".env") -> dict[str, str]:
    """Read ``KEY=VALUE`` lines from a dotenv file.

    Parameters
    ----------
    path : Path or str
        File to read. A missing file is not an error and gives an empty result,
        because the variables may equally well come from the real environment.

    Returns
    -------
    dict
        Parsed keys and values. Blank lines and ``#`` comments are skipped, an
        optional leading ``export`` is allowed, and surrounding single or double
        quotes are stripped.

    Notes
    -----
    Deliberately does not touch ``os.environ``: the caller decides precedence,
    and nothing here can leak a secret into a subprocess by accident.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def credentials_from_env(env_file: Path | str = ".env") -> tuple[str, str]:
    """Find Space-Track credentials, preferring the real environment.

    Returns
    -------
    (user, password) : tuple of str

    Raises
    ------
    AuthenticationError
        If either variable is missing, with a message that names the file and
        the variables but never their values.
    """
    from_file = load_dotenv(env_file)
    user = os.environ.get("SPACETRACK_USER") or from_file.get("SPACETRACK_USER", "")
    password = os.environ.get("SPACETRACK_PASSWORD") or from_file.get(
        "SPACETRACK_PASSWORD", ""
    )
    if not user or not password:
        missing = [
            name
            for name, value in (
                ("SPACETRACK_USER", user),
                ("SPACETRACK_PASSWORD", password),
            )
            if not value
        ]
        raise AuthenticationError(
            f"missing Space-Track credentials: {', '.join(missing)}. "
            f"Set them in the environment or in {env_file}, which is git-ignored."
        )
    return user, password


def _as_date_string(when: date | datetime | str) -> str:
    """Format a date the way Space-Track's EPOCH filter expects, YYYY-MM-DD."""
    if isinstance(when, str):
        return when
    return when.strftime("%Y-%m-%d")


@dataclass
class SpaceTrackClient:
    """A logged-in Space-Track session with throttling and a permanent cache.

    Attributes
    ----------
    user : str
        Account login.
    password : str
        Account password. Excluded from ``repr`` so it cannot reach a log or a
        traceback through an accidental print of the client.
    cache_dir : Path
        Where query responses are stored. Historical queries are immutable, so
        cached responses are reused forever unless ``refresh=True``.
    min_request_interval_s : float
        Minimum gap between outgoing requests.
    """

    user: str
    password: str = field(repr=False)
    cache_dir: Path = DEFAULT_CACHE_DIR
    min_request_interval_s: float = MIN_REQUEST_INTERVAL_S
    _opener: urllib.request.OpenerDirector | None = field(
        default=None, init=False, repr=False
    )
    _last_request_at: float = field(default=0.0, init=False, repr=False)

    @classmethod
    def from_env(
        cls, env_file: Path | str = ".env", cache_dir: Path = DEFAULT_CACHE_DIR
    ) -> SpaceTrackClient:
        """Build a client from ``SPACETRACK_USER`` and ``SPACETRACK_PASSWORD``."""
        user, password = credentials_from_env(env_file)
        return cls(user=user, password=password, cache_dir=cache_dir)

    # -- session ----------------------------------------------------------

    def login(self) -> None:
        """Open an authenticated session, reusing it for later queries.

        Raises
        ------
        AuthenticationError
            If the credentials are rejected. Space-Track signals this with
            HTTP 200 and ``{"Login":"Failed"}``, so the body is what decides.
        SpaceTrackError
            If the service cannot be reached.
        """
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cookie_jar),
            urllib.request.HTTPSHandler(context=ssl_context()),
        )
        opener.addheaders = [("User-Agent", USER_AGENT)]

        payload = urllib.parse.urlencode(
            {"identity": self.user, "password": self.password}
        ).encode("utf-8")

        self._throttle()
        try:
            with opener.open(LOGIN_URL, data=payload, timeout=_HTTP_TIMEOUT_S) as reply:
                body = reply.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as error:
            raise SpaceTrackError(
                f"could not reach Space-Track: {error.reason}"
            ) from error

        if "Failed" in body:
            raise AuthenticationError(
                "Space-Track rejected the credentials for user "
                f"{self.user!r}. Check SPACETRACK_USER and SPACETRACK_PASSWORD."
            )
        self._opener = opener

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.min_request_interval_s:
            time.sleep(self.min_request_interval_s - elapsed)
        self._last_request_at = time.monotonic()

    def _get(self, url: str) -> str:
        if self._opener is None:
            self.login()
        assert self._opener is not None

        self._throttle()
        try:
            with self._opener.open(url, timeout=_HTTP_TIMEOUT_S) as reply:
                return reply.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            if error.code == 401:
                raise AuthenticationError(
                    "Space-Track returned 401. The session expired or the "
                    "credentials are wrong."
                ) from error
            if error.code == 429:
                raise SpaceTrackError(
                    "Space-Track returned 429, too many requests. The documented "
                    "limits are 30 per minute and 300 per hour."
                ) from error
            raise SpaceTrackError(
                f"Space-Track returned HTTP {error.code} {error.reason}"
            ) from error
        except urllib.error.URLError as error:
            raise SpaceTrackError(
                f"could not reach Space-Track: {error.reason}"
            ) from error

    # -- queries ----------------------------------------------------------

    def gp_history_url(
        self,
        norad_id: int,
        start: date | datetime | str,
        stop: date | datetime | str,
    ) -> str:
        """Build the query URL for one object's element sets over a date range."""
        if norad_id >= TLE_FORMAT_MAX_NORAD_ID:
            raise ValueError(
                f"NORAD ID {norad_id} cannot be expressed in the legacy TLE output; "
                "request the 3le or json format instead"
            )
        return (
            f"{QUERY_URL}/class/gp_history"
            f"/NORAD_CAT_ID/{int(norad_id)}"
            f"/EPOCH/{_as_date_string(start)}--{_as_date_string(stop)}"
            f"/orderby/{urllib.parse.quote('EPOCH asc')}"
            "/format/tle"
        )

    def element_set_history(
        self,
        norad_id: int,
        start: date | datetime | str,
        stop: date | datetime | str,
        refresh: bool = False,
    ) -> list[TLE]:
        """Every element set published for one object within a date range.

        Parameters
        ----------
        norad_id : int
            Catalogue number.
        start, stop : date, datetime or str
            Inclusive epoch range. Strings are passed through, so Space-Track's
            own relative forms such as ``"now-30"`` also work.
        refresh : bool, optional
            Ignore the cached copy and query again. Off by default, because
            Space-Track asks that historical data be downloaded once and kept.

        Returns
        -------
        list of TLE
            In epoch order, oldest first. An object with no element sets in the
            range gives an empty list, which is a legitimate answer rather than
            an error.
        """
        url = self.gp_history_url(norad_id, start, stop)
        cache_file = self.cache_dir / f"{self._cache_key(url)}.tle"

        if not refresh and cache_file.is_file():
            text = cache_file.read_text(encoding="utf-8")
        else:
            text = self._get(url)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(text, encoding="utf-8")

        if not text.strip():
            return []
        records = parse_tle_text(text)
        return sorted(records, key=lambda record: record.epoch_utc)

    @staticmethod
    def _cache_key(url: str) -> str:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]

    def cached_queries(self) -> list[dict[str, str]]:
        """Describe what is already downloaded, for a quick inventory."""
        if not self.cache_dir.is_dir():
            return []
        return [
            {
                "file": path.name,
                "size_kb": f"{path.stat().st_size / 1024:.1f}",
                "records": str(sum(1 for line in path.open() if line.startswith("1 "))),
            }
            for path in sorted(self.cache_dir.glob("*.tle"))
        ]


def summarise_history(records: list[TLE]) -> dict[str, object]:
    """Quick statistics over a downloaded history, for the feasibility check.

    Returns the count, the epoch span, and the median and worst gap between
    consecutive element sets in hours. The gap is what limits how sharply any
    manoeuvre can be located in time.
    """
    if not records:
        return {"count": 0}

    epochs = [record.epoch_utc for record in records]
    gaps_h = [
        (later - earlier).total_seconds() / 3600.0
        for earlier, later in zip(epochs, epochs[1:], strict=False)
    ]
    gaps_h.sort()
    return {
        "count": len(records),
        "first_epoch": epochs[0].isoformat(),
        "last_epoch": epochs[-1].isoformat(),
        "span_days": round((epochs[-1] - epochs[0]).total_seconds() / 86400.0, 2),
        "median_gap_h": round(gaps_h[len(gaps_h) // 2], 3) if gaps_h else None,
        "max_gap_h": round(gaps_h[-1], 3) if gaps_h else None,
        "sets_per_day": round(len(records) / max((epochs[-1] - epochs[0]).days, 1), 2),
    }
