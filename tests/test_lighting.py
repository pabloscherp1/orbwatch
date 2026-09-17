"""Tests for Sun position and illumination.

The solar ephemeris is checked against astropy's full ephemeris, transformed to
TEME, offline using the IERS-B table bundled with astropy.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from orbwatch.access.lighting import (
    ASTRONOMICAL_UNIT_KM,
    is_sunlit,
    subsolar_point_rad,
    sun_position_km,
)
from orbwatch.catalog.frames import WGS84_EQUATORIAL_RADIUS_KM


def angle_between_rad(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Well-conditioned angle between row vectors, via atan2 of |a x b| and a.b."""
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    dot = np.einsum("ij,ij->i", a, b)
    return np.arctan2(cross, dot)


def test_sun_direction_agrees_with_astropy_to_a_minute_of_arc() -> None:
    """Measured on 2026-09-16: at most 32 arcsec over 2025 and 2026.

    The tolerance is 60 arcsec. The low-precision algorithm is quoted at about
    0.01 deg, and this also absorbs aberration and light time, which the
    algorithm ignores and astropy includes.
    """
    from astropy import units as u
    from astropy.coordinates import TEME, get_body
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False

    times = [
        datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(0, 580, 23)
    ]
    ours = sun_position_km(times)

    obstime = Time(times, scale="utc")
    reference = (
        get_body("sun", obstime)
        .transform_to(TEME(obstime=obstime))
        .cartesian.xyz.to_value(u.km)
        .T
    )

    assert np.rad2deg(angle_between_rad(ours, reference)).max() * 3600.0 < 60.0
    np.testing.assert_allclose(
        np.linalg.norm(ours, axis=1), np.linalg.norm(reference, axis=1), rtol=2e-4
    )


def test_sun_distance_stays_within_the_orbital_eccentricity_band() -> None:
    times = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=d) for d in range(366)]
    distance_au = np.linalg.norm(sun_position_km(times), axis=1) / ASTRONOMICAL_UNIT_KM
    assert distance_au.min() == pytest.approx(0.9833, abs=0.0005)
    assert distance_au.max() == pytest.approx(1.0167, abs=0.0005)


def test_object_between_earth_and_sun_is_sunlit() -> None:
    sun = np.array([[ASTRONOMICAL_UNIT_KM, 0.0, 0.0]])
    assert is_sunlit([[7000.0, 0.0, 0.0]], sun)[0]


def test_object_directly_behind_earth_is_in_shadow() -> None:
    sun = np.array([[ASTRONOMICAL_UNIT_KM, 0.0, 0.0]])
    assert not is_sunlit([[-7000.0, 0.0, 0.0]], sun)[0]


def test_object_behind_earth_but_outside_the_shadow_cylinder_is_sunlit() -> None:
    """Geostationary radius, behind the Earth but offset beyond its radius.
    This is why GEO objects are eclipsed only around the equinoxes."""
    sun = np.array([[ASTRONOMICAL_UNIT_KM, 0.0, 0.0]])
    offset = WGS84_EQUATORIAL_RADIUS_KM + 10.0
    assert is_sunlit([[-42164.0, 0.0, offset]], sun)[0]
    assert not is_sunlit([[-42164.0, 0.0, offset - 20.0]], sun)[0]


def test_subsolar_latitude_tracks_the_seasons() -> None:
    """Near the June solstice the Sun is overhead near the Tropic of Cancer,
    near the December solstice near the Tropic of Capricorn. Direction in the
    inertial frame is enough to check latitude, since latitude does not depend
    on Earth rotation."""
    june = sun_position_km(datetime(2026, 6, 21, 12, tzinfo=UTC))
    december = sun_position_km(datetime(2026, 12, 21, 12, tzinfo=UTC))

    june_lat, _ = subsolar_point_rad(june)
    december_lat, _ = subsolar_point_rad(december)

    assert np.rad2deg(june_lat[0]) == pytest.approx(23.44, abs=0.05)
    assert np.rad2deg(december_lat[0]) == pytest.approx(-23.44, abs=0.05)
