"""Tests for ground station visibility and pass prediction.

The reference is brute force: evaluate elevation every second over the same
window and read off the intervals above the mask directly. That shares the
frame and propagation code with the module under test, which those modules
test independently, but it shares none of the search logic, which is what is
being tested here.
"""

import warnings
from datetime import timedelta

import numpy as np
import pytest

from orbwatch.access.passes import (
    MEAN_EARTH_RADIUS_KM,
    GroundSite,
    find_passes,
    horizon_half_angle_rad,
    look_angles_over_time,
)
from orbwatch.catalog.frames import ecef_to_geodetic, teme_to_ecef_position
from orbwatch.catalog.propagate import propagate
from orbwatch.catalog.timescales import time_grid
from orbwatch.catalog.tle import TLE, parse_tle
from tests.test_tle import build_line1, build_line2

ZURICH = GroundSite("Zurich", 47.3769, 8.5417, 0.408)


@pytest.fixture
def iss_like() -> TLE:
    return parse_tle(build_line1(), build_line2())


def brute_force_intervals(tle, site, start, stop, mask_deg):
    times = list(time_grid(start, stop, 1.0))
    _, elevation, _ = look_angles_over_time(tle, site, times)
    above = elevation > np.deg2rad(mask_deg)
    intervals = []
    k = 0
    while k < len(times):
        if above[k]:
            j = k
            while j + 1 < len(times) and above[j + 1]:
                j += 1
            peak = k + int(np.argmax(elevation[k : j + 1]))
            intervals.append((times[k], times[j], elevation[peak]))
            k = j + 1
        else:
            k += 1
    return intervals


def test_passes_match_a_one_second_brute_force_scan(iss_like: TLE) -> None:
    start = iss_like.epoch_utc + timedelta(minutes=17)
    stop = start + timedelta(days=1)

    passes = find_passes(iss_like, ZURICH, start, stop)
    reference = brute_force_intervals(iss_like, ZURICH, start, stop, 0.0)

    assert len(passes) == len(reference) >= 3
    for found, (first_above, last_above, peak_rad) in zip(
        passes, reference, strict=True
    ):
        assert abs((found.aos_utc - first_above).total_seconds()) <= 1.0
        assert abs((found.los_utc - last_above).total_seconds()) <= 1.0
        assert found.max_elevation_rad >= peak_rad - 1e-6
        assert found.max_elevation_rad == pytest.approx(peak_rad, abs=np.deg2rad(0.01))


def test_refined_horizon_crossings_sit_on_the_mask(iss_like: TLE) -> None:
    start = iss_like.epoch_utc
    for mask_deg in (0.0, 10.0):
        passes = find_passes(
            iss_like,
            ZURICH,
            start,
            start + timedelta(days=1),
            min_elevation_deg=mask_deg,
        )
        assert passes
        for found in passes:
            for instant in (found.aos_utc, found.los_utc):
                _, elevation, _ = look_angles_over_time(iss_like, ZURICH, [instant])
                assert np.rad2deg(elevation[0]) == pytest.approx(mask_deg, abs=0.02)


def test_a_higher_mask_gives_shorter_passes_nested_inside_the_low_mask_ones(
    iss_like: TLE,
) -> None:
    start = iss_like.epoch_utc
    stop = start + timedelta(days=1)
    low = find_passes(iss_like, ZURICH, start, stop, min_elevation_deg=0.0)
    high = find_passes(iss_like, ZURICH, start, stop, min_elevation_deg=10.0)

    assert 0 < len(high) <= len(low)
    for inner in high:
        assert any(
            outer.aos_utc <= inner.aos_utc and inner.los_utc <= outer.los_utc
            for outer in low
        )


def test_window_opening_mid_pass_is_flagged_and_clipped(iss_like: TLE) -> None:
    first = find_passes(
        iss_like, ZURICH, iss_like.epoch_utc, iss_like.epoch_utc + timedelta(days=1)
    )[0]
    middle = first.aos_utc + (first.los_utc - first.aos_utc) / 2

    clipped = find_passes(iss_like, ZURICH, middle, middle + timedelta(hours=1))[0]

    assert clipped.starts_before_window
    assert clipped.aos_utc == middle
    assert not clipped.ends_after_window
    assert abs((clipped.los_utc - first.los_utc).total_seconds()) < 0.5


def test_geostationary_object_seen_from_beneath_is_one_pass_clipped_at_both_ends() -> (
    None
):
    geo = parse_tle(
        build_line1(bstar_field=" 00000+0", ndot_half_rev_per_day2=0.0),
        build_line2(
            inclination_deg=0.05,
            eccentricity_field="0002000",
            mean_motion_rev_per_day=1.00273791,
            revolution_number=1234,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ephemeris = propagate(geo, geo.epoch_utc)
    r_ecef = teme_to_ecef_position(ephemeris.r_teme_km, ephemeris.times_utc)
    _, lon, _ = ecef_to_geodetic(r_ecef)
    beneath = GroundSite("Beneath", 0.0, float(np.rad2deg(lon[0])) % 360.0, 0.0)

    passes = find_passes(geo, beneath, geo.epoch_utc, geo.epoch_utc + timedelta(days=1))

    assert len(passes) == 1
    assert passes[0].starts_before_window and passes[0].ends_after_window
    assert np.rad2deg(passes[0].max_elevation_rad) > 85.0


@pytest.mark.parametrize("alt_km", [420.0, 800.0, 20200.0, 35786.0])
@pytest.mark.parametrize("mask_deg", [0.0, 5.0, 10.0])
def test_horizon_half_angle_matches_the_law_of_sines(
    alt_km: float, mask_deg: float
) -> None:
    """Independent derivation. In the triangle Earth centre, site, object, the
    angle at the site is 90 deg + mask, so by the law of sines the nadir angle
    eta at the object satisfies sin(eta) = R cos(mask) / (R + h), and the three
    angles sum to 180 deg, giving lambda = 90 deg - mask - eta."""
    mask = np.deg2rad(mask_deg)
    radius = MEAN_EARTH_RADIUS_KM
    nadir = np.arcsin(radius * np.cos(mask) / (radius + alt_km))
    expected = np.pi / 2 - mask - nadir

    assert horizon_half_angle_rad(alt_km, mask) == pytest.approx(expected, abs=1e-12)


def test_horizon_half_angle_reference_values() -> None:
    assert np.rad2deg(horizon_half_angle_rad(420.0)) == pytest.approx(20.26, abs=0.01)
    assert np.rad2deg(horizon_half_angle_rad(35786.0)) == pytest.approx(81.31, abs=0.01)


def test_ground_site_validation() -> None:
    with pytest.raises(ValueError, match="latitude"):
        GroundSite("Nowhere", 91.0, 0.0)
    with pytest.raises(ValueError, match="longitude"):
        GroundSite("Nowhere", 0.0, 400.0)
    with pytest.raises(ValueError, match="altitude"):
        GroundSite("Orbit", 0.0, 0.0, 400.0)


def test_stop_before_start_is_rejected(iss_like: TLE) -> None:
    with pytest.raises(ValueError, match="after"):
        find_passes(iss_like, ZURICH, iss_like.epoch_utc, iss_like.epoch_utc)
