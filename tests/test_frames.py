"""Tests for frame conversions.

Three independent references are used:

- Hand-derivable geometry for the ellipsoid and for look angles.
- The ``sgp4`` library's ``gstime``, Vallado's reference GMST routine.
- astropy's full IAU TEME to ITRS transformation, which includes UT1 and polar
  motion. This module ignores both, so the test asserts that the residual is
  *explained* by UT1 minus UTC, rather than just that it is small.
"""

import warnings
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sgp4.propagation import gstime

from orbwatch.catalog.frames import (
    EARTH_ROTATION_RATE_RAD_S,
    WGS84_EQUATORIAL_RADIUS_KM,
    WGS84_POLAR_RADIUS_KM,
    ecef_to_geodetic,
    geodetic_to_ecef,
    gmst_rad,
    look_angles,
    teme_to_ecef,
    teme_to_ecef_position,
)
from orbwatch.catalog.propagate import propagate
from orbwatch.catalog.timescales import julian_date_split_array
from orbwatch.catalog.tle import parse_tle
from tests.test_tle import build_line1, build_line2

# --------------------------------------------------------------------------
# GMST
# --------------------------------------------------------------------------


def test_gmst_matches_vallado_gstime() -> None:
    times = [
        datetime(2026, 9, 16, tzinfo=UTC) + timedelta(hours=h) for h in range(0, 48, 5)
    ]
    jd_midnight, fraction = julian_date_split_array(times)
    reference = np.array(
        [gstime(j + f) for j, f in zip(jd_midnight, fraction, strict=True)]
    )
    np.testing.assert_allclose(gmst_rad(times), reference, atol=1e-8)


def test_gmst_advances_one_sidereal_rate_per_second() -> None:
    """Earth turns 360 deg relative to the stars in one sidereal day, so GMST
    advances at the Earth rotation rate."""
    start = datetime(2026, 9, 16, 3, tzinfo=UTC)
    later = start + timedelta(seconds=600)
    advance = (gmst_rad(later)[0] - gmst_rad(start)[0]) % (2 * np.pi)
    assert advance / 600.0 == pytest.approx(EARTH_ROTATION_RATE_RAD_S, rel=1e-6)


# --------------------------------------------------------------------------
# Geodetic
# --------------------------------------------------------------------------


def test_equator_prime_meridian_is_on_the_x_axis() -> None:
    np.testing.assert_allclose(
        geodetic_to_ecef(0.0, 0.0, 0.0),
        [WGS84_EQUATORIAL_RADIUS_KM, 0.0, 0.0],
        atol=1e-9,
    )


def test_north_pole_sits_at_the_polar_radius() -> None:
    np.testing.assert_allclose(
        geodetic_to_ecef(np.pi / 2, 0.0, 0.0),
        [0.0, 0.0, WGS84_POLAR_RADIUS_KM],
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("lat_deg", "lon_deg", "alt_km"),
    [
        (0.0, 0.0, 0.0),
        (47.3769, 8.5417, 0.408),
        (-33.9, 18.4, 0.0),
        (89.9999, -120.0, 400.0),
        (-90.0, 0.0, 800.0),
        (0.05, -75.2, 35786.0),
        (51.6, 179.999, 420.0),
    ],
)
def test_geodetic_round_trip(lat_deg: float, lon_deg: float, alt_km: float) -> None:
    r = geodetic_to_ecef(np.deg2rad(lat_deg), np.deg2rad(lon_deg), alt_km)
    lat, lon, alt = ecef_to_geodetic(r)

    assert lat[0] == pytest.approx(np.deg2rad(lat_deg), abs=1e-11)
    assert alt[0] == pytest.approx(alt_km, abs=1e-6)
    if abs(lat_deg) < 90.0:
        assert np.cos(lon[0] - np.deg2rad(lon_deg)) == pytest.approx(1.0, abs=1e-12)


def test_geodetic_latitude_exceeds_geocentric_by_about_0_19_deg_at_45() -> None:
    """Geodetic latitude exceeds geocentric by roughly f sin(2 lat), which peaks
    near 45 deg at about 0.19 deg, or some 21 km of surface position."""
    r = geodetic_to_ecef(np.deg2rad(45.0), 0.0, 0.0)
    geocentric_deg = np.rad2deg(np.arctan2(r[2], r[0]))
    assert 45.0 - geocentric_deg == pytest.approx(0.1924, abs=0.001)


# --------------------------------------------------------------------------
# Look angles
# --------------------------------------------------------------------------


def _local_to_ecef(
    site_lat: float, site_lon: float, east: float, north: float, up: float
):
    """Place a point at a known east/north/up offset from a site."""
    sin_lat, cos_lat = np.sin(site_lat), np.cos(site_lat)
    sin_lon, cos_lon = np.sin(site_lon), np.cos(site_lon)
    east_hat = np.array([-sin_lon, cos_lon, 0.0])
    north_hat = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up_hat = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
    site = geodetic_to_ecef(site_lat, site_lon, 0.0)
    return site + east * east_hat + north * north_hat + up * up_hat


@pytest.mark.parametrize(
    ("east", "north", "up", "az_deg", "el_deg"),
    [
        (0.0, 1000.0, 0.0, 0.0, 0.0),
        (1000.0, 0.0, 0.0, 90.0, 0.0),
        (0.0, -1000.0, 0.0, 180.0, 0.0),
        (-1000.0, 0.0, 0.0, 270.0, 0.0),
        (0.0, 1000.0, 1000.0, 0.0, 45.0),
        (1e-9, 0.0, 400.0, None, 90.0),
    ],
)
def test_look_angles_for_known_local_offsets(
    east: float, north: float, up: float, az_deg: float | None, el_deg: float
) -> None:
    site_lat, site_lon = np.deg2rad(47.3769), np.deg2rad(8.5417)
    target = _local_to_ecef(site_lat, site_lon, east, north, up)

    azimuth, elevation, range_km = look_angles(target, site_lat, site_lon, 0.0)

    assert np.rad2deg(elevation[0]) == pytest.approx(el_deg, abs=1e-9)
    assert range_km[0] == pytest.approx(np.sqrt(east**2 + north**2 + up**2), rel=1e-12)
    if az_deg is not None:
        difference = (np.rad2deg(azimuth[0]) - az_deg + 180.0) % 360.0 - 180.0
        assert difference == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# TEME to Earth-fixed, against astropy's full transformation
# --------------------------------------------------------------------------


def test_teme_to_ecef_agrees_with_astropy_to_within_the_ut1_residual() -> None:
    """Our chain uses UTC for UT1 and ignores polar motion. astropy uses both.

    The expected residual is therefore about r * omega_earth * |UT1 - UTC| plus
    a few tens of metres of polar motion. Measured on 2026-09-16 for this date:
    29 to 34 m, with UT1 - UTC = 0.062 s. The assertion bounds the residual by
    that physical explanation, so the test fails if something other than the
    two documented simplifications creeps in.

    The date is inside the IERS-B table bundled with astropy, and automatic
    download is disabled, so this runs offline.
    """
    from astropy import units as u
    from astropy.coordinates import (
        ITRS,
        TEME,
        CartesianDifferential,
        CartesianRepresentation,
    )
    from astropy.time import Time
    from astropy.utils import iers

    iers.conf.auto_download = False

    tle = parse_tle(build_line1(epoch_year="26", epoch_day=74.25), build_line2())
    start = datetime(2026, 3, 15, 6, 0, tzinfo=UTC)
    times = [start + timedelta(minutes=m) for m in range(0, 180, 7)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ephemeris = propagate(tle, times)

    r_ecef, v_ecef = teme_to_ecef(ephemeris.r_teme_km, ephemeris.v_teme_km_s, times)

    obstime = Time(times, scale="utc")
    teme = TEME(
        CartesianRepresentation(
            ephemeris.r_teme_km.T * u.km,
            differentials=CartesianDifferential(ephemeris.v_teme_km_s.T * u.km / u.s),
        ),
        obstime=obstime,
    )
    itrs = teme.transform_to(ITRS(obstime=obstime))
    reference_r = itrs.cartesian.xyz.to_value(u.km).T
    reference_v = itrs.cartesian.differentials["s"].d_xyz.to_value(u.km / u.s).T

    radius_km = np.linalg.norm(ephemeris.r_teme_km, axis=1)
    ut1_residual_km = (
        radius_km * EARTH_ROTATION_RATE_RAD_S * np.abs(obstime.delta_ut1_utc)
    )
    polar_motion_allowance_km = 0.05

    position_error_km = np.linalg.norm(r_ecef - reference_r, axis=1)
    assert np.all(position_error_km < ut1_residual_km + polar_motion_allowance_km)

    velocity_error_km_s = np.linalg.norm(v_ecef - reference_v, axis=1)
    assert np.all(velocity_error_km_s < 1e-4)


def test_earth_fixed_velocity_loses_the_rotation_term() -> None:
    """A point fixed on the equator, expressed in TEME, must have zero
    Earth-fixed velocity once the transport term is removed."""
    time = datetime(2026, 9, 16, 12, tzinfo=UTC)
    theta = gmst_rad(time)[0]
    r_ecef = np.array([WGS84_EQUATORIAL_RADIUS_KM, 0.0, 0.0])

    # Rotate the Earth-fixed point back into TEME and give it the velocity of
    # the rotating Earth at that point.
    r_teme = np.array([r_ecef[0] * np.cos(theta), r_ecef[0] * np.sin(theta), 0.0])
    v_teme = np.cross([0.0, 0.0, EARTH_ROTATION_RATE_RAD_S], r_teme)

    r_out, v_out = teme_to_ecef(r_teme, v_teme, time)
    np.testing.assert_allclose(r_out[0], r_ecef, atol=1e-9)
    np.testing.assert_allclose(v_out[0], [0.0, 0.0, 0.0], atol=1e-12)


def test_position_only_conversion_matches_full_conversion() -> None:
    time = datetime(2026, 9, 16, 12, tzinfo=UTC)
    r_teme = np.array([[6800.0, 1200.0, -300.0]])
    v_teme = np.array([[-1.0, 7.4, 1.0]])
    full, _ = teme_to_ecef(r_teme, v_teme, time)
    np.testing.assert_allclose(teme_to_ecef_position(r_teme, time), full, atol=1e-12)


def test_mismatched_shapes_are_rejected() -> None:
    time = datetime(2026, 9, 16, tzinfo=UTC)
    with pytest.raises(ValueError, match="expected"):
        teme_to_ecef(np.zeros((2, 3)), np.zeros((2, 3)), time)
