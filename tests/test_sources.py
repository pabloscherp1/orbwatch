"""Tests for the catalogue fetching layer.

The network itself is stubbed out here. What is being tested is the caching
policy and the failure handling, which are the parts that decide whether this
module is polite to a free public service and honest when something goes wrong.
The live fetch is exercised by the network-marked test in ``test_tle.py``.
"""

from pathlib import Path

import pytest

from orbwatch.catalog import sources
from orbwatch.catalog.sources import CatalogFetchError, fetch_celestrak_tle
from tests.test_tle import build_line1, build_line2


def make_response(norad_id: int = 25544, name: str = "ISS (ZARYA)") -> str:
    return "\r\n".join(
        [name, build_line1(norad_id=norad_id), build_line2(norad_id=norad_id), ""]
    )


@pytest.fixture
def stub_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the HTTP call with a stub, recording every URL requested."""
    requested: list[str] = []

    def fake_get(url: str, timeout_s: float = 30.0) -> str:
        requested.append(url)
        return make_response()

    monkeypatch.setattr(sources, "http_get_text", fake_get)
    return requested


def test_fetch_parses_the_response_and_keeps_the_name(
    stub_network: list[str], tmp_path: Path
) -> None:
    tle = fetch_celestrak_tle(25544, cache_dir=tmp_path)

    assert tle.norad_id == 25544
    assert tle.name == "ISS (ZARYA)"
    assert len(stub_network) == 1
    assert "CATNR=25544" in stub_network[0]


def test_a_second_fetch_is_served_from_cache(
    stub_network: list[str], tmp_path: Path
) -> None:
    """A repeated request inside the cache window must not hit the network."""
    fetch_celestrak_tle(25544, cache_dir=tmp_path)
    fetch_celestrak_tle(25544, cache_dir=tmp_path)

    assert len(stub_network) == 1, "the second call should have been cached"
    assert len(list(tmp_path.glob("*.tle"))) == 1


def test_zero_cache_age_forces_a_fresh_request(
    stub_network: list[str], tmp_path: Path
) -> None:
    fetch_celestrak_tle(25544, cache_dir=tmp_path)
    fetch_celestrak_tle(25544, cache_dir=tmp_path, max_cache_age_s=0.0)

    assert len(stub_network) == 2


def test_different_objects_do_not_share_a_cache_entry(
    stub_network: list[str], tmp_path: Path
) -> None:
    fetch_celestrak_tle(25544, cache_dir=tmp_path)
    fetch_celestrak_tle(43013, cache_dir=tmp_path)

    assert len(stub_network) == 2
    assert len(list(tmp_path.glob("*.tle"))) == 2


def test_unknown_object_is_reported_clearly_and_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Celestrak answers an unknown catalogue number with HTTP 404.

    Observed on 2026-09-17, with the body "No GP data found". The fetcher must
    report a missing object distinctly from an outage, which is what lets the
    interface return 404 rather than 502, and must not cache the failure.
    """

    def not_found(url: str, timeout_s: float = 30.0) -> str:
        raise CatalogFetchError(f"{url} returned HTTP 404 Not Found", status=404)

    monkeypatch.setattr(sources, "http_get_text", not_found)

    with pytest.raises(CatalogFetchError, match="no current element set") as caught:
        fetch_celestrak_tle(99999, cache_dir=tmp_path)
    assert caught.value.status == 404
    assert not list(tmp_path.glob("*.tle"))


def test_an_outage_is_not_mistaken_for_a_missing_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def outage(url: str, timeout_s: float = 30.0) -> str:
        raise CatalogFetchError(
            f"{url} returned HTTP 503 Service Unavailable", status=503
        )

    monkeypatch.setattr(sources, "http_get_text", outage)

    with pytest.raises(CatalogFetchError, match="503") as caught:
        fetch_celestrak_tle(25544, cache_dir=tmp_path)
    assert caught.value.status == 503


def test_empty_response_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sources, "http_get_text", lambda url, timeout_s=30.0: "   \n")

    with pytest.raises(CatalogFetchError, match="no current element set"):
        fetch_celestrak_tle(99999, cache_dir=tmp_path)


def test_cache_path_is_deterministic_and_url_specific() -> None:
    first = sources._cache_path(Path("/tmp"), "https://example.invalid/a")
    again = sources._cache_path(Path("/tmp"), "https://example.invalid/a")
    other = sources._cache_path(Path("/tmp"), "https://example.invalid/b")

    assert first == again
    assert first != other
    assert first.suffix == ".tle"
