"""Impulsive manoeuvres between near-circular orbits.

The building blocks of a transfer: raising or lowering an orbit, turning its
plane, and combining the two. Every function assumes impulsive burns, which is
accurate for chemical thrusters whose burns last a small fraction of an orbit.

Units: km, km/s, seconds, radians.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbwatch.catalog.elements import MU_EARTH_KM3_S2

_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0
_SPLIT_TOLERANCE = 1e-10


def circular_speed_km_s(r_km: float, mu_km3_s2: float = MU_EARTH_KM3_S2) -> float:
    return float(np.sqrt(mu_km3_s2 / r_km))


def period_s(a_km: float, mu_km3_s2: float = MU_EARTH_KM3_S2) -> float:
    return float(2.0 * np.pi * np.sqrt(a_km**3 / mu_km3_s2))


def plane_change_dv_km_s(speed_km_s: float, angle_rad: float) -> float:
    """Speed change that turns a velocity through ``angle_rad`` at constant size.

    2 v sin(angle / 2). At 7.5 km/s one degree costs 131 m/s, which is why
    plane changes dominate low-orbit budgets and are avoided where possible.
    """
    return float(2.0 * speed_km_s * np.sin(abs(angle_rad) / 2.0))


def orbit_plane_angle_rad(
    inclination1_rad: float,
    raan1_rad: float,
    inclination2_rad: float,
    raan2_rad: float,
) -> float:
    """Angle between two orbit planes, from their inclinations and nodes."""
    cos_angle = np.cos(inclination1_rad) * np.cos(inclination2_rad) + np.sin(
        inclination1_rad
    ) * np.sin(inclination2_rad) * np.cos(raan2_rad - raan1_rad)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


@dataclass(frozen=True)
class Transfer:
    """A two-burn transfer between circular orbits.

    Attributes
    ----------
    dv1_km_s, dv2_km_s : float
        Speed change of the departure and arrival burns.
    time_of_flight_s : float
        Half the period of the transfer ellipse.
    plane_split : float
        Share of the plane change done at the departure burn, 0 to 1.
    """

    dv1_km_s: float
    dv2_km_s: float
    time_of_flight_s: float
    plane_split: float = 0.0

    @property
    def dv_km_s(self) -> float:
        return self.dv1_km_s + self.dv2_km_s


def _burn_km_s(v_before: float, v_after: float, angle_rad: float) -> float:
    """Speed change between two velocities with an angle between them."""
    return float(
        np.sqrt(v_before**2 + v_after**2 - 2.0 * v_before * v_after * np.cos(angle_rad))
    )


def hohmann(
    r1_km: float,
    r2_km: float,
    plane_change_rad: float = 0.0,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> Transfer:
    """Hohmann transfer, with an optional plane change split optimally.

    Parameters
    ----------
    r1_km, r2_km : float
        Radii of the departure and arrival circular orbits.
    plane_change_rad : float, optional
        Angle between the two orbit planes. It is shared between the two burns
        in whatever proportion minimises the total, found by golden-section
        search on a convex function. Most of it goes at the slower burn, at the
        larger radius, where turning the velocity is cheaper.
    mu_km3_s2 : float, optional

    Returns
    -------
    Transfer
    """
    a_transfer = 0.5 * (r1_km + r2_km)
    v1 = circular_speed_km_s(r1_km, mu_km3_s2)
    v2 = circular_speed_km_s(r2_km, mu_km3_s2)
    v_depart = float(np.sqrt(mu_km3_s2 * (2.0 / r1_km - 1.0 / a_transfer)))
    v_arrive = float(np.sqrt(mu_km3_s2 * (2.0 / r2_km - 1.0 / a_transfer)))
    angle = abs(plane_change_rad)

    def total(split: float) -> float:
        return _burn_km_s(v1, v_depart, split * angle) + _burn_km_s(
            v_arrive, v2, (1.0 - split) * angle
        )

    low, high = 0.0, 1.0
    while high - low > _SPLIT_TOLERANCE:
        a = high - _GOLDEN * (high - low)
        b = low + _GOLDEN * (high - low)
        if total(a) < total(b):
            high = b
        else:
            low = a
    split = 0.5 * (low + high) if angle > 0.0 else 0.0
    return Transfer(
        dv1_km_s=_burn_km_s(v1, v_depart, split * angle),
        dv2_km_s=_burn_km_s(v_arrive, v2, (1.0 - split) * angle),
        time_of_flight_s=0.5 * period_s(a_transfer, mu_km3_s2),
        plane_split=split,
    )


def synodic_period_s(
    a1_km: float, a2_km: float, mu_km3_s2: float = MU_EARTH_KM3_S2
) -> float:
    """Time for one orbit to lap another: how long the relative phase takes to
    come round. Waiting at most this long sets any phase at no speed cost."""
    n1 = 2.0 * np.pi / period_s(a1_km, mu_km3_s2)
    n2 = 2.0 * np.pi / period_s(a2_km, mu_km3_s2)
    if n1 == n2:
        return float("inf")
    return float(2.0 * np.pi / abs(n1 - n2))
