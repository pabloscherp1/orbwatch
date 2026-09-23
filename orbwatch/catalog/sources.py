"""Fetching element sets from public catalogue services.

Network and disk live here so that :mod:`orbwatch.catalog.tle` stays pure and
offline-testable.

Two services, with different roles:

- **Celestrak** serves the *current* element set for an object, without
  authentication. Use it for "where is this object now".
- **Space-Track** serves the *historical* record, and requires an account. Use
  it for "what has this object been doing", which is what manoeuvre detection
  needs. Handled in a separate function because the credential and rate-limit
  story is entirely different.

Responses are cached to disk. These are free public services and one of them is
run on a shoestring, so re-requesting the same element set on every run is both
rude and slow. Element sets are regenerated a few times a day at most, so a
two-hour default cache costs nothing in freshness.

**Do not commit cached data.** ``data/cache/`` is in ``.gitignore`` and should
stay there. Redistribution of catalogue data is governed by terms this project
does not attempt to interpret.
"""

from __future__ import annotations

import hashlib
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi

from orbwatch.catalog.tle import TLE, parse_tle_text

CELESTRAK_GP_URL: str = "https://celestrak.org/NORAD/elements/gp.php"

DEFAULT_CACHE_DIR: Path = Path("data/cache")

DEFAULT_MAX_CACHE_AGE_S: float = 7200.0
"""Two hours. Element sets are regenerated a few times a day at most."""

USER_AGENT: str = "orbwatch/0.0.1 (open-source mission design toolchain)"

_HTTP_TIMEOUT_S: float = 30.0


class CatalogFetchError(RuntimeError):
    """Raised when a catalogue service cannot be reached or refuses a request.

    ``status`` carries the HTTP status code when the service answered with an
    error, and is None for network-level failures.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def ssl_context() -> ssl.SSLContext:
    """An SSL context backed by the certifi CA bundle.

    The python.org macOS installer does not populate a certificate store, so
    stdlib HTTPS fails there with CERTIFICATE_VERIFY_FAILED until something
    supplies one. Pinning certifi explicitly makes the behaviour identical on
    every machine instead of depending on how Python was installed.
    """
    return ssl.create_default_context(cafile=certifi.where())


def _cache_path(cache_dir: Path, url: str) -> Path:
    """Deterministic cache filename for a URL.

    The digest keeps query strings, which carry the catalogue number and the
    date range, from turning into unusable filenames.
    """
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.tle"


def _read_cache(path: Path, max_age_s: float) -> str | None:
    """Return cached text if it exists and is fresh enough, otherwise None."""
    if not path.is_file():
        return None
    if time.time() - path.stat().st_mtime > max_age_s:
        return None
    return path.read_text(encoding="utf-8")


def http_get_text(url: str, timeout_s: float = _HTTP_TIMEOUT_S) -> str:
    """Fetch a URL and decode the body as text.

    Parameters
    ----------
    url : str
        Absolute URL.
    timeout_s : float, optional
        Socket timeout in seconds.

    Returns
    -------
    str
        Decoded response body.

    Raises
    ------
    CatalogFetchError
        On any HTTP or network failure, with the original error attached as the
        exception cause so the traceback still shows what actually happened.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(
            request, timeout=timeout_s, context=ssl_context()
        ) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise CatalogFetchError(
            f"{url} returned HTTP {error.code} {error.reason}", status=error.code
        ) from error
    except urllib.error.URLError as error:
        raise CatalogFetchError(f"could not reach {url}: {error.reason}") from error


def fetch_celestrak_tle(
    norad_id: int,
    cache_dir: Path | None = None,
    max_cache_age_s: float = DEFAULT_MAX_CACHE_AGE_S,
) -> TLE:
    """Fetch the current element set for one catalogue object from Celestrak.

    Parameters
    ----------
    norad_id : int
        NORAD catalogue number.
    cache_dir : Path or None, optional
        Directory for cached responses. Defaults to ``data/cache``. Pass a
        temporary directory in tests to avoid polluting the working cache.
    max_cache_age_s : float, optional
        Serve from cache if the cached copy is younger than this. Set to 0 to
        force a fresh request.

    Returns
    -------
    TLE
        The current element set, with the object name from the title line.

    Raises
    ------
    CatalogFetchError
        If the service is unreachable, or if it has no element set for this
        catalogue number. The latter is reported with the message "no current
        element set" and status 404, so callers can tell a missing object from
        an outage.

    Notes
    -----
    This is the *current* element set only. Celestrak does not serve history,
    so manoeuvre detection needs Space-Track instead.
    """
    cache_dir = DEFAULT_CACHE_DIR if cache_dir is None else cache_dir
    url = f"{CELESTRAK_GP_URL}?CATNR={int(norad_id)}&FORMAT=tle"

    path = _cache_path(cache_dir, url)
    text = _read_cache(path, max_cache_age_s)

    missing = CatalogFetchError(
        f"Celestrak has no current element set for NORAD ID {norad_id}. "
        "The object may be unknown, decayed, or not publicly catalogued.",
        status=404,
    )

    if text is None:
        # Observed on 2026-09-17: Celestrak answers an unknown catalogue number
        # with HTTP 404 and the body "No GP data found". Error responses are
        # never cached, so a missing object is re-checked on the next request.
        try:
            text = http_get_text(url)
        except CatalogFetchError as error:
            if error.status == 404:
                raise missing from error
            raise
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    if not text.strip():
        raise missing

    records = parse_tle_text(text)
    if len(records) != 1:
        raise CatalogFetchError(
            f"expected exactly one element set for NORAD ID {norad_id}, "
            f"got {len(records)}"
        )
    return records[0]
