"""Earth's oblateness, J2, as a tool: turning an orbit plane for free.

Earth's equatorial bulge makes every inclined orbit's plane precess about the
polar axis. The rate depends on altitude and inclination, so two spacecraft in
slightly different orbits drift apart in node, or together. A mission whose
target sits in a different plane can therefore wait in a chosen drift orbit
until J2 has turned its plane into the target's, instead of turning it with
propellant: a degree of node costs over 100 m/s by thrust at these altitudes.

Secular rates only, first order in J2. Units: km, radians, seconds.
"""

from __future__ import annotations

import numpy as np

from orbwatch.catalog.elements import MU_EARTH_KM3_S2

J2: float = 1.08262668e-3
"""Earth's second zonal harmonic, EGM96."""

EARTH_RADIUS_KM: float = 6378.137

TROPICAL_YEAR_S: float = 365.2422 * 86400.0
SUN_SYNCHRONOUS_RATE_RAD_S: float = 2.0 * np.pi / TROPICAL_YEAR_S
"""Nodal rate that keeps an orbit plane at a fixed angle to the Sun."""


def nodal_rate_rad_s(
    a_km: float,
    inclination_rad: float,
    eccentricity: float = 0.0,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> float:
    """Secular rate of the right ascension of the ascending node.

    -3/2 J2 (R/p)^2 n cos i. Negative, westward, for prograde orbits and
    positive for retrograde ones; about +0.9856 deg/day for a sun-synchronous
    orbit.
    """
    n = np.sqrt(mu_km3_s2 / a_km**3)
    p = a_km * (1.0 - eccentricity**2)
    return float(-1.5 * J2 * (EARTH_RADIUS_KM / p) ** 2 * n * np.cos(inclination_rad))


def sun_synchronous_inclination_rad(
    a_km: float, eccentricity: float = 0.0, mu_km3_s2: float = MU_EARTH_KM3_S2
) -> float:
    """Inclination at which the node turns once a year, following the Sun.

    Raises
    ------
    ValueError
        Above about 5,970 km altitude, where J2 is too weak for any inclination.
    """
    n = np.sqrt(mu_km3_s2 / a_km**3)
    p = a_km * (1.0 - eccentricity**2)
    cos_i = -SUN_SYNCHRONOUS_RATE_RAD_S / (1.5 * J2 * (EARTH_RADIUS_KM / p) ** 2 * n)
    if abs(cos_i) > 1.0:
        raise ValueError("no sun-synchronous inclination at this altitude")
    return float(np.arccos(cos_i))
