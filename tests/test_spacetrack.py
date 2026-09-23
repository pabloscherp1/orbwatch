"""Tests for the Space-Track history client.

Nothing here touches the network or reads real credentials. What is under test
is the part that is easy to get wrong and expensive to get wrong: judging
authentication from the response body rather than its status, never leaking the
password, caching historical queries permanently, and throttling requests to
stay inside the documented limits.
"""

import urllib.error
import urllib.request
from datetime import date, timedelta

import pytest

from orbwatch.catalog import spacetrack
from orbwatch.catalog.spacetrack import (
    AuthenticationError,
    SpaceTrackClient,
    SpaceTrackError,
    credentials_from_env,
    load_dotenv,
    summarise_history,
)
from orbwatch.catalog.tle import parse_tle_text
from tests.test_tle import build_line1, build_line2

PASSWORD = "correct horse battery staple"


def history_text(count: int, first_day: float = 250.0, step_days: float = 0.5) -> str:
    """Synthetic element set history, two-line format, one record per epoch."""
    lines = []
    for index in range(count):
        day = first_day + index * step_days
        lines.append(build_line1(epoch_year="26", epoch_day=day))
        lines.append(build_line2(revolution_number=58000 + index))
    return "\n".join(lines) + "\n"


@pytest.fixture
def client(tmp_path) -> SpaceTrackClient:
    return SpaceTrackClient(
        user="pablo", password=PASSWORD, cache_dir=tmp_path, min_request_interval_s=0.0
    )


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------


def test_dotenv_parsing(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n"
        "\n"
        "SPACETRACK_USER=pablo@example.com\n"
        'export SPACETRACK_PASSWORD="quoted secret"\n'
        "MALFORMED\n"
        "EMPTY=\n"
    )
    values = load_dotenv(path)

    assert values["SPACETRACK_USER"] == "pablo@example.com"
    assert values["SPACETRACK_PASSWORD"] == "quoted secret"
    assert values["EMPTY"] == ""
    assert "MALFORMED" not in values


def test_dotenv_missing_file_is_not_an_error(tmp_path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_dotenv_does_not_touch_the_process_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    path = tmp_path / ".env"
    path.write_text("SPACETRACK_USER=from_file\n")

    load_dotenv(path)

    import os

    assert "SPACETRACK_USER" not in os.environ


def test_environment_beats_the_dotenv_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text("SPACETRACK_USER=from_file\nSPACETRACK_PASSWORD=from_file\n")
    monkeypatch.setenv("SPACETRACK_USER", "from_env")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "from_env")

    assert credentials_from_env(path) == ("from_env", "from_env")


def test_missing_credentials_name_the_variables_but_not_a_value(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    path = tmp_path / ".env"
    path.write_text("SPACETRACK_USER=pablo\n")

    with pytest.raises(AuthenticationError) as caught:
        credentials_from_env(path)

    message = str(caught.value)
    assert "SPACETRACK_PASSWORD" in message
    assert "SPACETRACK_USER" not in message.split(":")[1]
    assert "git-ignored" in message


def test_password_never_appears_in_the_repr(client: SpaceTrackClient) -> None:
    assert PASSWORD not in repr(client)
    assert "pablo" in repr(client)


# --------------------------------------------------------------------------
# Login, judged on the body
# --------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeOpener:
    def __init__(self, body: str = "", error: Exception | None = None) -> None:
        self.body = body
        self.error = error
        self.calls: list[tuple[str, bytes | None]] = []
        self.addheaders: list[tuple[str, str]] = []

    def open(self, url, data=None, timeout=None):  # noqa: ANN001, matches urllib
        self.calls.append((url, data))
        if self.error is not None:
            raise self.error
        return FakeResponse(self.body)


def install_opener(monkeypatch, opener: FakeOpener) -> FakeOpener:
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **k: opener)
    return opener


def test_wrong_password_is_caught_even_though_the_status_is_200(
    client: SpaceTrackClient, monkeypatch
) -> None:
    """Space-Track answers a failed login with HTTP 200 and {"Login":"Failed"}."""
    opener = install_opener(monkeypatch, FakeOpener(body='{"Login":"Failed"}'))

    with pytest.raises(AuthenticationError) as caught:
        client.login()

    assert PASSWORD not in str(caught.value)
    assert "SPACETRACK_PASSWORD" in str(caught.value)
    assert opener.calls[0][0] == spacetrack.LOGIN_URL


def test_successful_login_posts_the_documented_fields(
    client: SpaceTrackClient, monkeypatch
) -> None:
    opener = install_opener(monkeypatch, FakeOpener(body='""'))

    client.login()

    url, data = opener.calls[0]
    assert url == spacetrack.LOGIN_URL
    assert b"identity=pablo" in data
    assert b"password=" in data


def test_unreachable_service_is_reported_as_such(
    client: SpaceTrackClient, monkeypatch
) -> None:
    install_opener(monkeypatch, FakeOpener(error=urllib.error.URLError("offline")))

    with pytest.raises(SpaceTrackError, match="could not reach"):
        client.login()


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------


def test_query_url_matches_the_documented_form(client: SpaceTrackClient) -> None:
    url = client.gp_history_url(25544, date(2025, 9, 20), date(2026, 9, 20))

    assert url.startswith(f"{spacetrack.QUERY_URL}/class/gp_history")
    assert "/NORAD_CAT_ID/25544" in url
    assert "/EPOCH/2025-09-20--2026-09-20" in url
    assert "/orderby/EPOCH%20asc" in url
    assert url.endswith("/format/tle")


def test_relative_date_strings_pass_through(client: SpaceTrackClient) -> None:
    assert "/EPOCH/now-30--now" in client.gp_history_url(25544, "now-30", "now")


def test_catalogue_numbers_too_large_for_the_tle_format_are_refused(
    client: SpaceTrackClient,
) -> None:
    with pytest.raises(ValueError, match="legacy TLE output"):
        client.gp_history_url(100000, "now-1", "now")


def test_history_is_parsed_sorted_and_cached(
    client: SpaceTrackClient, monkeypatch
) -> None:
    calls: list[str] = []

    def fake_get(url: str) -> str:
        calls.append(url)
        return history_text(5)

    monkeypatch.setattr(client, "_get", fake_get)

    records = client.element_set_history(25544, "2026-09-01", "2026-09-10")
    assert len(calls) == 1
    assert len(records) == 5
    assert [r.epoch_utc for r in records] == sorted(r.epoch_utc for r in records)

    again = client.element_set_history(25544, "2026-09-01", "2026-09-10")
    assert len(calls) == 1, "a second identical query must come from the cache"
    assert [r.epoch_utc for r in again] == [r.epoch_utc for r in records]

    client.element_set_history(25544, "2026-09-01", "2026-09-10", refresh=True)
    assert len(calls) == 2, "refresh=True must bypass the cache"


def test_different_ranges_do_not_share_a_cache_entry(
    client: SpaceTrackClient, monkeypatch
) -> None:
    monkeypatch.setattr(client, "_get", lambda url: history_text(2))

    client.element_set_history(25544, "2026-01-01", "2026-02-01")
    client.element_set_history(25544, "2026-02-01", "2026-03-01")

    assert len(list(client.cache_dir.glob("*.tle"))) == 2
    assert len(client.cached_queries()) == 2


def test_an_object_with_no_history_gives_an_empty_list(
    client: SpaceTrackClient, monkeypatch
) -> None:
    monkeypatch.setattr(client, "_get", lambda url: "\n")
    assert client.element_set_history(25544, "2026-01-01", "2026-01-02") == []


@pytest.mark.parametrize(
    ("status", "expected", "fragment"),
    [
        (401, AuthenticationError, "401"),
        (429, SpaceTrackError, "30 per minute"),
        (503, SpaceTrackError, "503"),
    ],
)
def test_http_errors_are_translated(
    client: SpaceTrackClient, monkeypatch, status: int, expected: type, fragment: str
) -> None:
    error = urllib.error.HTTPError(
        spacetrack.QUERY_URL, status, "boom", hdrs=None, fp=None
    )
    opener = FakeOpener(error=error)
    client._opener = opener

    with pytest.raises(expected, match=fragment):
        client._get(spacetrack.QUERY_URL)


def test_requests_are_throttled_to_the_documented_rate(monkeypatch, tmp_path) -> None:
    """The client must wait between requests rather than hammering the service."""
    slept: list[float] = []
    clock = {"now": 1000.0}

    monkeypatch.setattr(spacetrack.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(spacetrack.time, "sleep", lambda s: slept.append(s))

    client = SpaceTrackClient(
        user="pablo", password=PASSWORD, cache_dir=tmp_path, min_request_interval_s=3.0
    )
    client._opener = FakeOpener(body="")

    client._get(spacetrack.QUERY_URL)
    assert slept == [], "the first request should not wait"

    clock["now"] += 0.5
    client._get(spacetrack.QUERY_URL)
    assert slept == [pytest.approx(2.5)], "the second must wait out the interval"


# --------------------------------------------------------------------------
# Summary used by the feasibility check
# --------------------------------------------------------------------------


def test_summarise_history_reports_span_and_gaps() -> None:
    records = parse_tle_text(history_text(9, first_day=250.0, step_days=0.5))
    summary = summarise_history(records)

    assert summary["count"] == 9
    assert summary["span_days"] == pytest.approx(4.0, abs=0.01)
    assert summary["median_gap_h"] == pytest.approx(12.0, abs=0.01)
    assert summary["max_gap_h"] == pytest.approx(12.0, abs=0.01)
    assert summary["sets_per_day"] == pytest.approx(2.25, abs=0.01)


def test_summarise_empty_history() -> None:
    assert summarise_history([]) == {"count": 0}


# --------------------------------------------------------------------------
# Live check, skipped unless credentials exist
# --------------------------------------------------------------------------


@pytest.mark.network
def test_live_query_returns_recent_iss_element_sets(tmp_path) -> None:
    try:
        client = SpaceTrackClient.from_env(cache_dir=tmp_path)
    except AuthenticationError as error:
        pytest.skip(f"no Space-Track credentials: {error}")

    stop = date.today()
    start = stop - timedelta(days=3)
    records = client.element_set_history(25544, start, stop)

    assert records, "the ISS should have element sets in any recent 3 day window"
    assert all(record.norad_id == 25544 for record in records)
    assert records[0].epoch_utc <= records[-1].epoch_utc
