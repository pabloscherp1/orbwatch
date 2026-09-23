"""Tests for the tracker interface backend.

The payload functions are tested directly. The server is started on a free
port with a synthetic element set provider, so none of this touches the
network, and the static file handling is checked for the two things that
matter in any hand-rolled file server: it serves what it should, and it refuses
to serve anything outside its directory.
"""

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import date, timedelta

import numpy as np
import pytest

from orbwatch.access.passes import GroundSite
from orbwatch.catalog.sources import CatalogFetchError
from orbwatch.catalog.spacetrack import AuthenticationError
from orbwatch.catalog.tle import TLE, parse_tle
from orbwatch.gui import api
from orbwatch.gui.server import create_server
from tests.test_tle import build_line1, build_line2

ZURICH = GroundSite("Zurich", 47.3769, 8.5417, 0.408)


@pytest.fixture
def iss_like() -> TLE:
    return parse_tle(build_line1(), build_line2(), name="ISS-LIKE")


# --------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------


def test_track_payload_has_parallel_arrays_and_is_strict_json(iss_like: TLE) -> None:
    payload = api.track(
        iss_like, ZURICH, iss_like.epoch_utc, before_s=600, after_s=1200, step_s=30
    )
    samples = payload["samples"]
    lengths = {name: len(values) for name, values in samples.items()}

    assert set(lengths.values()) == {61}
    json.dumps(payload, allow_nan=False)
    assert samples["t_unix_ms"][20] == int(iss_like.epoch_utc.timestamp() * 1000)


def test_track_payload_values_are_physically_sensible(iss_like: TLE) -> None:
    samples = api.track(
        iss_like, ZURICH, iss_like.epoch_utc, before_s=0, after_s=5400, step_s=60
    )["samples"]

    assert all(380.0 < a < 460.0 for a in samples["alt_km"])
    assert all(-51.8 < lat < 51.8 for lat in samples["lat_deg"])
    assert all(19.0 < f < 22.0 for f in samples["footprint_deg"])
    assert all(
        v == (e > 0.0)
        for v, e in zip(samples["in_view"], samples["elevation_deg"], strict=True)
    )
    assert all(
        not o or (s and v)
        for o, s, v in zip(
            samples["optically_visible"],
            samples["sunlit"],
            samples["in_view"],
            strict=True,
        )
    )


def test_failed_samples_become_null_not_nan() -> None:
    decaying = parse_tle(
        build_line1(bstar_field=" 50000-1", ndot_half_rev_per_day2=0.01),
        build_line2(mean_motion_rev_per_day=16.2),
    )
    payload = api.track(
        decaying, ZURICH, decaying.epoch_utc, before_s=0, after_s=172800, step_s=3600
    )
    samples = payload["samples"]

    assert samples["valid"][0] is True
    assert samples["valid"][-1] is False
    assert samples["lat_deg"][-1] is None
    json.dumps(payload, allow_nan=False)


def test_track_flags_a_stale_element_set(iss_like: TLE) -> None:
    far = iss_like.epoch_utc + timedelta(days=40)
    payload = api.track(iss_like, ZURICH, far, before_s=0, after_s=60, step_s=60)
    assert payload["stale"] is True

    near = api.track(
        iss_like, ZURICH, iss_like.epoch_utc, before_s=0, after_s=60, step_s=60
    )
    assert near["stale"] is False


def test_track_rejects_oversized_windows(iss_like: TLE) -> None:
    with pytest.raises(ValueError, match="samples"):
        api.track(
            iss_like,
            ZURICH,
            iss_like.epoch_utc,
            before_s=86400,
            after_s=86400,
            step_s=1,
        )
    with pytest.raises(ValueError, match="step_s"):
        api.track(
            iss_like, ZURICH, iss_like.epoch_utc, before_s=0, after_s=60, step_s=0.5
        )


def test_passes_payload_is_ordered_and_consistent(iss_like: TLE) -> None:
    payload = api.passes(iss_like, ZURICH, iss_like.epoch_utc, hours=24)
    found = payload["passes"]

    assert len(found) >= 3
    for entry in found:
        assert (
            entry["aos_unix_ms"]
            <= entry["max_elevation_unix_ms"]
            <= entry["los_unix_ms"]
        )
    starts = [entry["aos_unix_ms"] for entry in found]
    assert starts == sorted(starts)


def test_object_summary_reports_mean_elements_and_age(iss_like: TLE) -> None:
    summary = api.object_summary(iss_like, iss_like.epoch_utc + timedelta(days=2))
    assert summary["name"] == "ISS-LIKE"
    assert summary["age_days"] == pytest.approx(2.0)
    assert summary["nominal_period_min"] == pytest.approx(
        1440.0 / 15.49122235, abs=1e-3
    )
    assert summary["mean_elements"]["inclination_deg"] == pytest.approx(51.6311)


def leo_history(days: float = 50.0, burn_km: float = 1.2, seed: int = 30) -> list[TLE]:
    """ISS-like element sets every 4 to 8 hours: drag decay, 5 m of fit noise,
    and one reboost of ``burn_km`` halfway through."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(4.0, 8.0, int(days * 24 / 4)) / 24.0)
    t = t[t < days]
    a = 6800.0 - 0.055 * t + rng.normal(0.0, 0.005, t.size)
    a[t > days / 2] += burn_km
    n_rev_day = np.sqrt(398600.8 / a**3) * 86400.0 / (2.0 * np.pi)
    return [
        parse_tle(
            build_line1(epoch_year="26", epoch_day=10.0 + day),
            build_line2(mean_motion_rev_per_day=float(n)),
        )
        for day, n in zip(t, n_rev_day, strict=True)
    ]


def test_behaviour_payload_finds_the_reboost_and_is_strict_json() -> None:
    payload = api.behaviour(leo_history())
    json.dumps(payload, allow_nan=False)

    series = payload["series"]
    assert len({len(values) for values in series.values()}) == 1
    assert payload["regime"] == "other"
    assert payload["natural"] is None

    burns = [e for e in payload["events"] if e["status"] == "manoeuvre"]
    assert len(burns) == 1
    assert burns[0]["kind"] == "in_plane"
    assert burns[0]["delta_a_km"] == pytest.approx(1.2, abs=0.03)
    assert burns[0]["start_unix_ms"] < burns[0]["end_unix_ms"]

    detector = payload["detector"]["in_plane"]
    assert detector["unit"] == "m"
    assert len(detector["t_unix_ms"]) == len(detector["magnitude"])
    assert sum(detector["flagged"]) >= 1
    assert 3.0 < detector["sigma"] < 8.0, "5 m of fit noise, reported in metres"
    assert payload["budgets"]["in_plane"]["closure"] == pytest.approx(1.0, abs=0.1)


def test_budget_says_whether_burns_stand_out_from_the_scatter() -> None:
    budget = api.behaviour(leo_history())["budgets"]["in_plane"]

    assert budget["burns"] == 1
    expected = 0.5 * np.sqrt(398600.8 / 6800.0) * 1000.0 * 1.2 / 6800.0
    assert budget["typical_burn_m_s"] == pytest.approx(expected, rel=0.05)
    assert 0.001 < budget["scatter_per_gap_m_s"] < 0.005, "5 m of fit noise in LEO"
    assert budget["typical_burn_m_s"] / budget["scatter_per_gap_m_s"] > 100
    assert budget["required_per_gap_m_s"] > 0


@pytest.mark.parametrize(
    ("observed_deg", "required_m_s", "control"),
    [
        (0.02, 47.0, "held"),
        (0.98, 4.5, "free"),
        (0.50, 20.0, "partial"),
    ],
)
def test_plane_control_separates_held_free_and_partial(
    observed_deg: float, required_m_s: float, control: str
) -> None:
    natural = np.deg2rad(0.9)
    observed = np.deg2rad(observed_deg)
    assert api.plane_control(natural, observed, required_m_s, 3074.7) == control


def test_behaviour_refuses_a_history_too_short_to_measure_noise() -> None:
    with pytest.raises(ValueError, match="at least"):
        api.behaviour(leo_history()[:5])


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------


class FakeHistory:
    """Stands in for Space-Track, and counts how often it is asked."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, date, date]] = []

    def __call__(self, norad_id: int, start: date, stop: date) -> list[TLE]:
        self.calls.append((norad_id, start, stop))
        if norad_id == 25544:
            return leo_history()
        if norad_id == 40000:
            raise AuthenticationError(
                "missing Space-Track credentials: SPACETRACK_USER, "
                "SPACETRACK_PASSWORD. Set them in the environment or in .env"
            )
        if norad_id == 22222:
            return leo_history()[:5]
        return []


@pytest.fixture
def history() -> FakeHistory:
    return FakeHistory()


@pytest.fixture
def server_url(iss_like: TLE, history: FakeHistory) -> Iterator[str]:
    def provider(norad_id: int) -> TLE:
        if norad_id == 25544:
            return iss_like
        raise CatalogFetchError(
            f"Celestrak has no current element set for NORAD ID {norad_id}."
        )

    server = create_server(
        "127.0.0.1", 0, tle_provider=provider, history_provider=history
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def get(url: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return (
                response.status,
                response.read(),
                response.headers.get("Content-Type", ""),
            )
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "")


def test_health_endpoint(server_url: str) -> None:
    status, body, content_type = get(f"{server_url}/api/health")
    assert status == 200
    assert content_type.startswith("application/json")
    assert json.loads(body)["status"] == "ok"


def test_track_endpoint_round_trip(server_url: str, iss_like: TLE) -> None:
    t = iss_like.epoch_utc.isoformat().replace("+00:00", "Z")
    status, body, _ = get(
        f"{server_url}/api/track?norad=25544&t={t}&before=0&after=600&step=60"
        "&lat=47.3769&lon=8.5417&alt=0.408&site=Zurich"
    )
    assert status == 200
    payload = json.loads(body)
    assert len(payload["samples"]["t_unix_ms"]) == 11
    assert payload["site"]["name"] == "Zurich"


@pytest.mark.parametrize(
    ("query", "status", "fragment"),
    [
        ("/api/object?norad=abc", 400, "positive integer"),
        ("/api/object", 400, "missing"),
        ("/api/object?norad=99999", 404, "no current element set"),
        ("/api/track?norad=25544&lat=95&lon=0", 400, "latitude"),
        ("/api/track?norad=25544&lat=47&lon=8&t=2026-09-16T08:00:00", 400, "timezone"),
        ("/api/track?norad=25544&lat=47&lon=8&step=0.1", 400, "step_s"),
        ("/api/nope", 404, "no such endpoint"),
        ("/api/behaviour?norad=25544&days=3", 400, "days"),
        ("/api/behaviour?norad=25544&days=90.5", 400, "whole number"),
        ("/api/behaviour?norad=25544&sigmas=1", 400, "sigmas"),
        ("/api/behaviour?norad=40000", 503, "credentials"),
        ("/api/behaviour?norad=11111", 404, "no element sets"),
        ("/api/behaviour?norad=22222", 422, "at least"),
    ],
)
def test_bad_requests_get_clear_json_errors(
    server_url: str, query: str, status: int, fragment: str
) -> None:
    code, body, content_type = get(f"{server_url}{query}")
    assert code == status
    assert content_type.startswith("application/json")
    assert fragment in json.loads(body)["error"]


def test_behaviour_endpoint_analyses_once_and_then_serves_from_memory(
    server_url: str, history: FakeHistory
) -> None:
    url = f"{server_url}/api/behaviour?norad=25544&days=90"
    first = get(url)
    second = get(url)

    assert first[0] == second[0] == 200
    assert first[1] == second[1]
    assert len(history.calls) == 1
    norad_id, start, stop = history.calls[0]
    assert norad_id == 25544
    assert (stop - start).days == 90
    payload = json.loads(first[1])
    assert [e["status"] for e in payload["events"]].count("manoeuvre") == 1

    get(f"{server_url}/api/behaviour?norad=25544&days=90&sigmas=6")
    assert len(history.calls) == 2, "a different threshold is a different analysis"


def test_index_page_is_served(server_url: str) -> None:
    status, body, content_type = get(f"{server_url}/")
    assert status == 200
    assert content_type.startswith("text/html")
    assert b"ORBWATCH" in body


def test_files_outside_the_static_directory_are_not_served(server_url: str) -> None:
    for path in ("/../server.py", "/..%2fserver.py", "/%2e%2e/%2e%2e/pyproject.toml"):
        status, body, _ = get(f"{server_url}{path}")
        assert status == 404
        assert b"def main" not in body and b"[project]" not in body


def test_directory_listing_is_disabled(server_url: str) -> None:
    status, _, _ = get(f"{server_url}/vendor/")
    assert status == 404
