"""Reference frame conversions: TEME, Earth-fixed, geodetic and topocentric.

The chain implemented here is the one Vallado gives for SGP4 output:

    TEME  --rotate by GMST about z-->  PEF  --polar motion-->  ITRF

with two documented simplifications, chosen because the whole of ORBWATCH works
at the kilometre level that TLE accuracy allows, not the metre level:

- **UTC is used in place of UT1** for the Earth rotation angle. UT1 minus UTC is
  kept within 0.9 s by leap seconds, which is at most about 0.5 km of position
  error at LEO radius, well inside the error of the element set itself.
- **Polar motion is ignored**, so PEF is used as Earth-fixed. Polar motion is a
  few tenths of an arcsecond, which is of order 10 m at LEO radius.

Both are checked against astropy's full IAU transformation, which includes UT1
and polar motion, in ``tests/test_frames.py``. That test is also the answer to
whether the GMST rotation alone is good enough for a ground track: it is, to
about 1.5 arcseconds.

**What is deliberately not here.** No J2000 or GCRF conversion, because nothing
in the pipeline needs one yet, and no LVLH or RIC frames, which arrive with the
rendezvous work.

Units: km, km/s, radians. Geodetic coordinates are on the WGS-84 ellipsoid.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike, NDArray

from orbwatch.catalog.timescales import julian_date_split_array

EARTH_ROTATION_RATE_RAD_S: float = 7.292115146706979e-5
"""Nominal Earth rotation rate, rad/s. Textbook value, as used by Vallado."""

WGS84_EQUATORIAL_RADIUS_KM: float = 6378.137
WGS84_FLATTENING: float = 1.0 / 298.257223563
WGS84_ECCENTRICITY_SQUARED: float = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
WGS84_POLAR_RADIUS_KM: float = WGS84_EQUATORIAL_RADIUS_KM * (1.0 - WGS84_FLATTENING)

J2000_JULIAN_DATE: float = 2451545.0

_GEODETIC_MAX_ITERATIONS: int = 10
_GEODETIC_TOLERANCE_RAD: float = 1e-13


def _as_times(times_utc: datetime | Sequence[datetime]) -> Sequence[datetime]:
    return (times_utc,) if isinstance(times_utc, datetime) else times_utc


def gmst_rad(times_utc: datetime | Sequence[datetime]) -> NDArray[np.float64]:
    """Greenwich mean sidereal time, IAU 1982 model.

    Parameters
    ----------
    times_utc : datetime or sequence of datetime
        Timezone-aware instants. UTC is used in place of UT1, see module notes.

    Returns
    -------
    ndarray, shape (N,)
        GMST in [0, 2*pi), radians.

    Notes
    -----
    This is the same polynomial as Vallado's ``gstime`` routine, which the
    ``sgp4`` library uses internally and which ``tests/test_frames.py`` checks
    against. It is the angle the Earth has turned through relative to the mean
    equinox, and it is exactly the angle that takes TEME to Earth-fixed.

    The split Julian date is subtracted from J2000 before the two parts are
    added, which keeps the fractional day at full precision.
    """
    jd_midnight, day_fraction = julian_date_split_array(_as_times(times_utc))
    centuries = ((jd_midnight - J2000_JULIAN_DATE) + day_fraction) / 36525.0

    gmst_seconds = (
        -6.2e-6 * centuries**3
        + 0.093104 * centuries**2
        + (876600.0 * 3600.0 + 8640184.812866) * centuries
        + 67310.54841
    )
    # One second of sidereal time is 15 arcseconds, which is 1/240 of a degree.
    return np.deg2rad(gmst_seconds / 240.0) % (2.0 * np.pi)


def _rotation_about_z(angle_rad: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stack of frame rotations R3(angle), shape (N, 3, 3).

    R3 rotates the *axes* by ``angle`` about z, so it maps a vector's
    components from the old frame into the new one.
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    zeros = np.zeros_like(angle_rad)
    ones = np.ones_like(angle_rad)
    return np.stack(
        [
            np.stack([cos_a, sin_a, zeros], axis=-1),
            np.stack([-sin_a, cos_a, zeros], axis=-1),
            np.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )


def teme_to_ecef(
    r_teme_km: ArrayLike,
    v_teme_km_s: ArrayLike,
    times_utc: datetime | Sequence[datetime],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotate TEME states into the Earth-fixed frame.

    Parameters
    ----------
    r_teme_km : array_like, shape (N, 3) or (3,)
        Positions in TEME, km.
    v_teme_km_s : array_like, shape (N, 3) or (3,)
        Velocities in TEME, km/s.
    times_utc : datetime or sequence of datetime
        One instant per row.

    Returns
    -------
    r_ecef_km : ndarray, shape (N, 3)
        Earth-fixed position, km.
    v_ecef_km_s : ndarray, shape (N, 3)
        Earth-fixed velocity, km/s, *relative to the rotating Earth*.

    Notes
    -----
    Position is a pure rotation. Velocity is not: the Earth-fixed frame rotates,
    so the velocity seen from it loses the transport term,

        v_ecef = R3(gmst) v_teme - omega_earth x r_ecef.

    Forgetting that term is a classic error. It is about 0.5 km/s at LEO, which
    is enormous, and it makes Earth-fixed speeds look plausible but wrong.
    """
    times = _as_times(times_utc)
    r_teme = np.atleast_2d(np.asarray(r_teme_km, dtype=float))
    v_teme = np.atleast_2d(np.asarray(v_teme_km_s, dtype=float))
    if r_teme.shape != (len(times), 3) or v_teme.shape != (len(times), 3):
        raise ValueError(
            f"expected ({len(times)}, 3) position and velocity arrays for "
            f"{len(times)} times, got {r_teme.shape} and {v_teme.shape}"
        )

    rotation = _rotation_about_z(gmst_rad(times))
    r_ecef = np.einsum("nij,nj->ni", rotation, r_teme)
    omega = np.array([0.0, 0.0, EARTH_ROTATION_RATE_RAD_S])
    v_ecef = np.einsum("nij,nj->ni", rotation, v_teme) - np.cross(omega, r_ecef)
    return r_ecef, v_ecef


def teme_to_ecef_position(
    r_teme_km: ArrayLike,
    times_utc: datetime | Sequence[datetime],
) -> NDArray[np.float64]:
    """Rotate TEME positions into the Earth-fixed frame, km, shape (N, 3)."""
    times = _as_times(times_utc)
    r_teme = np.atleast_2d(np.asarray(r_teme_km, dtype=float))
    if r_teme.shape != (len(times), 3):
        raise ValueError(f"expected ({len(times)}, 3) positions, got {r_teme.shape}")
    return np.einsum("nij,nj->ni", _rotation_about_z(gmst_rad(times)), r_teme)


def geodetic_to_ecef(
    lat_rad: ArrayLike,
    lon_rad: ArrayLike,
    alt_km: ArrayLike,
) -> NDArray[np.float64]:
    """Geodetic latitude, longitude and altitude on WGS-84 to Earth-fixed.

    Parameters
    ----------
    lat_rad, lon_rad : array_like
        Geodetic latitude and longitude, radians.
    alt_km : array_like
        Height above the ellipsoid, km.

    Returns
    -------
    ndarray, shape (..., 3)
        Earth-fixed position, km.
    """
    lat = np.asarray(lat_rad, dtype=float)
    lon = np.asarray(lon_rad, dtype=float)
    alt = np.asarray(alt_km, dtype=float)

    sin_lat = np.sin(lat)
    # Prime vertical radius of curvature: distance from the surface to the polar
    # axis along the ellipsoid normal.
    prime_vertical_km = WGS84_EQUATORIAL_RADIUS_KM / np.sqrt(
        1.0 - WGS84_ECCENTRICITY_SQUARED * sin_lat**2
    )
    x = (prime_vertical_km + alt) * np.cos(lat) * np.cos(lon)
    y = (prime_vertical_km + alt) * np.cos(lat) * np.sin(lon)
    z = (prime_vertical_km * (1.0 - WGS84_ECCENTRICITY_SQUARED) + alt) * sin_lat
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(
    r_ecef_km: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Earth-fixed position to geodetic latitude, longitude and altitude.

    Parameters
    ----------
    r_ecef_km : array_like, shape (N, 3) or (3,)
        Earth-fixed position, km.

    Returns
    -------
    lat_rad : ndarray, shape (N,)
        Geodetic latitude, radians, in [-pi/2, pi/2].
    lon_rad : ndarray, shape (N,)
        Longitude, radians, in (-pi, pi].
    alt_km : ndarray, shape (N,)
        Height above the WGS-84 ellipsoid, km.

    Notes
    -----
    Latitude and altitude are coupled through the ellipsoid, so there is no
    simple closed form and this iterates. It converges to well below a
    millimetre in a handful of iterations for any orbit.

    Altitude uses ``h = p cos(lat) + z sin(lat) - a sqrt(1 - e^2 sin^2 lat)``
    rather than the more common ``h = p / cos(lat) - N``, because the common
    form divides by ``cos(lat)`` and falls apart over the poles.
    """
    r = np.atleast_2d(np.asarray(r_ecef_km, dtype=float))
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    e2 = WGS84_ECCENTRICITY_SQUARED
    a = WGS84_EQUATORIAL_RADIUS_KM

    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))

    for _ in range(_GEODETIC_MAX_ITERATIONS):
        prime_vertical = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        lat_next = np.arctan2(z + e2 * prime_vertical * np.sin(lat), p)
        converged = np.all(np.abs(lat_next - lat) < _GEODETIC_TOLERANCE_RAD)
        lat = lat_next
        if converged:
            break

    alt = p * np.cos(lat) + z * np.sin(lat) - a * np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return lat, lon, alt


def look_angles(
    r_ecef_km: ArrayLike,
    site_lat_rad: float,
    site_lon_rad: float,
    site_alt_km: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Azimuth, elevation and range of targets seen from a ground site.

    Parameters
    ----------
    r_ecef_km : array_like, shape (N, 3) or (3,)
        Target positions, Earth-fixed, km.
    site_lat_rad, site_lon_rad : float
        Site geodetic latitude and longitude, radians.
    site_alt_km : float
        Site height above the WGS-84 ellipsoid, km.

    Returns
    -------
    azimuth_rad : ndarray, shape (N,)
        Clockwise from true north, in [0, 2*pi).
    elevation_rad : ndarray, shape (N,)
        Above the local horizontal plane, in [-pi/2, pi/2].
    range_km : ndarray, shape (N,)
        Straight-line distance from the site, km.

    Notes
    -----
    "Up" is the ellipsoid normal, not the direction away from the Earth's
    centre. They differ by up to about 0.19 deg, which is enough to shift a
    pass's rise and set times by seconds.

    Elevation comes from ``arctan2(up, horizontal)`` rather than
    ``arcsin(up / range)``, for the same conditioning reason as in
    :mod:`orbwatch.catalog.elements`: ``arcsin`` loses precision near 90 deg.
    """
    targets = np.atleast_2d(np.asarray(r_ecef_km, dtype=float))
    site = geodetic_to_ecef(site_lat_rad, site_lon_rad, site_alt_km)
    line_of_sight = targets - site

    sin_lat, cos_lat = np.sin(site_lat_rad), np.cos(site_lat_rad)
    sin_lon, cos_lon = np.sin(site_lon_rad), np.cos(site_lon_rad)
    east_hat = np.array([-sin_lon, cos_lon, 0.0])
    north_hat = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    up_hat = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])

    east = line_of_sight @ east_hat
    north = line_of_sight @ north_hat
    up = line_of_sight @ up_hat

    azimuth = np.arctan2(east, north) % (2.0 * np.pi)
    elevation = np.arctan2(up, np.hypot(east, north))
    range_km = np.linalg.norm(line_of_sight, axis=1)
    return azimuth, elevation, range_km
