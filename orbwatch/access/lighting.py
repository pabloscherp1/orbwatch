"""Sun position and illumination geometry.

Lighting decides what an optical sensor can see: a satellite is only visible to
a telescope when it is sunlit and the telescope is in darkness. That single
constraint is why optical ground surveillance of LEO only works in the twilight
hours, and it is one of the reasons geometry, not sensor quality, sets the
limits of ground-based SSA.

The solar ephemeris is the low-precision algorithm from the Astronomical Almanac
as given by Vallado. It is a few hundredths of a degree from a full ephemeris,
checked against astropy in ``tests/test_lighting.py``, which is far below
anything that matters for illumination or for drawing a terminator.

Units: km, radians.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike, NDArray

from orbwatch.catalog.frames import J2000_JULIAN_DATE, WGS84_EQUATORIAL_RADIUS_KM
from orbwatch.catalog.timescales import julian_date_split_array

ASTRONOMICAL_UNIT_KM: float = 149597870.7
"""IAU 2012 definition, exact."""

CIVIL_TWILIGHT_SUN_ELEVATION_RAD: float = np.deg2rad(-6.0)
"""Sun this far below the horizon marks the end of civil twilight."""


def _as_times(times_utc: datetime | Sequence[datetime]) -> Sequence[datetime]:
    return (times_utc,) if isinstance(times_utc, datetime) else times_utc


def sun_position_km(
    times_utc: datetime | Sequence[datetime],
) -> NDArray[np.float64]:
    """Geocentric Sun position, low-precision, km, shape (N, 3).

    Parameters
    ----------
    times_utc : datetime or sequence of datetime
        Timezone-aware instants.

    Returns
    -------
    ndarray, shape (N, 3)
        Geocentric position of the Sun on the mean equator and equinox of date,
        km. At this precision that is interchangeable with TEME, which differs
        from it only by the equation of the equinoxes, about a second of arc.

    Notes
    -----
    Mean longitude and mean anomaly advance linearly in Julian centuries, the
    equation of centre adds the two largest periodic terms, and the Sun is
    placed on the ecliptic and rotated onto the equator by the obliquity. UTC
    stands in for both UT1 and TDB, which is a minute-level time error and so a
    few thousandths of a degree of solar motion.
    """
    jd_midnight, fraction = julian_date_split_array(_as_times(times_utc))
    centuries = ((jd_midnight - J2000_JULIAN_DATE) + fraction) / 36525.0

    mean_longitude_deg = 280.460 + 36000.771 * centuries
    mean_anomaly_rad = np.deg2rad(357.5291092 + 35999.05034 * centuries)
    ecliptic_longitude_rad = np.deg2rad(
        mean_longitude_deg
        + 1.914666471 * np.sin(mean_anomaly_rad)
        + 0.019994643 * np.sin(2.0 * mean_anomaly_rad)
    )
    distance_au = (
        1.000140612
        - 0.016708617 * np.cos(mean_anomaly_rad)
        - 0.000139589 * np.cos(2.0 * mean_anomaly_rad)
    )
    obliquity_rad = np.deg2rad(23.439291 - 0.0130042 * centuries)

    distance_km = distance_au * ASTRONOMICAL_UNIT_KM
    return np.stack(
        [
            distance_km * np.cos(ecliptic_longitude_rad),
            distance_km * np.cos(obliquity_rad) * np.sin(ecliptic_longitude_rad),
            distance_km * np.sin(obliquity_rad) * np.sin(ecliptic_longitude_rad),
        ],
        axis=-1,
    )


def is_sunlit(
    r_teme_km: ArrayLike,
    r_sun_km: ArrayLike,
) -> NDArray[np.bool_]:
    """Whether each position is in sunlight, cylindrical shadow model.

    Parameters
    ----------
    r_teme_km : array_like, shape (N, 3)
        Object positions, km.
    r_sun_km : array_like, shape (N, 3)
        Sun positions in the same frame, km.

    Returns
    -------
    ndarray of bool, shape (N,)

    Notes
    -----
    The Earth's shadow is modelled as a cylinder of the equatorial radius
    extending away from the Sun. The real shadow is a cone with a penumbra, so
    this model gets eclipse entry and exit wrong by a few seconds in LEO. That
    is fine for deciding whether an object is observable during a pass, and not
    fine for a power or thermal analysis.
    """
    r = np.atleast_2d(np.asarray(r_teme_km, dtype=float))
    sun = np.atleast_2d(np.asarray(r_sun_km, dtype=float))
    sun_hat = sun / np.linalg.norm(sun, axis=1, keepdims=True)

    along_sun = np.einsum("ij,ij->i", r, sun_hat)
    distance_from_shadow_axis = np.linalg.norm(r - along_sun[:, None] * sun_hat, axis=1)
    return (along_sun > 0.0) | (distance_from_shadow_axis > WGS84_EQUATORIAL_RADIUS_KM)


def subsolar_point_rad(
    r_sun_ecef_km: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Latitude and longitude of the point directly beneath the Sun.

    Returns geocentric latitude, which for a direction at a distance of one
    astronomical unit is indistinguishable from geodetic. Radians.
    """
    sun = np.atleast_2d(np.asarray(r_sun_ecef_km, dtype=float))
    lat = np.arctan2(sun[:, 2], np.hypot(sun[:, 0], sun[:, 1]))
    lon = np.arctan2(sun[:, 1], sun[:, 0])
    return lat, lon
