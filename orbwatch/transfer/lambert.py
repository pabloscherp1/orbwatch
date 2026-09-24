"""Lambert's problem: the orbit through two positions in a given time.

Given where a spacecraft is, where it has to be and how long it has, Lambert's
problem returns the velocities at both ends. Every two-impulse transfer, from
an orbit raise to a far-range rendezvous, is a Lambert problem.

Universal-variable formulation after Curtis, algorithm 5.2, solved by
bisection on the universal variable z rather than Newton's method: the time of
flight increases monotonically with z over a single revolution, so bisection
cannot diverge, and the few hundred evaluations it needs cost nothing here.

Units: km, km/s, seconds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from orbwatch.catalog.elements import MU_EARTH_KM3_S2
from orbwatch.transfer.kepler import stumpff_c, stumpff_s

_Z_MAX: float = 4.0 * np.pi**2
"""Upper end of z for a single revolution: the transfer becomes a full ellipse."""

_BISECTION_ITERATIONS: int = 200
_DEGENERATE_SINE: float = 1e-8


def lambert(
    r1_km: ArrayLike,
    r2_km: ArrayLike,
    tof_s: float,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
    normal: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Velocities at both ends of the single-revolution transfer.

    Parameters
    ----------
    r1_km, r2_km : array_like, shape (3,)
        Departure and arrival positions, inertial frame.
    tof_s : float
        Time of flight, seconds, positive.
    mu_km3_s2 : float, optional
    normal : array_like, shape (3,), optional
        Direction the transfer's angular momentum should point along, which
        decides between the short and the long way round. Defaults to +z, the
        usual prograde convention. Pass the departure orbit's angular momentum
        for retrograde orbits, such as sun-synchronous ones.

    Returns
    -------
    v1_km_s, v2_km_s : ndarray, shape (3,)

    Raises
    ------
    ValueError
        For a non-positive time of flight, or a transfer angle of 0 or 180 deg,
        where the transfer plane is undefined.
    """
    if tof_s <= 0.0:
        raise ValueError("time of flight must be positive")
    r1 = np.asarray(r1_km, dtype=float)
    r2 = np.asarray(r2_km, dtype=float)
    r1n = float(np.linalg.norm(r1))
    r2n = float(np.linalg.norm(r2))
    reference = np.array([0.0, 0.0, 1.0]) if normal is None else np.asarray(normal)

    cos_angle = float(np.clip(np.dot(r1, r2) / (r1n * r2n), -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if np.dot(np.cross(r1, r2), reference) < 0.0:
        angle = 2.0 * np.pi - angle
    if abs(np.sin(angle)) < _DEGENERATE_SINE:
        raise ValueError(
            "transfer angle of 0 or 180 deg: the transfer plane is undefined"
        )
    a_const = np.sin(angle) * np.sqrt(r1n * r2n / (1.0 - np.cos(angle)))
    root_mu = np.sqrt(mu_km3_s2)

    def y(z: float) -> float:
        return r1n + r2n + a_const * (z * stumpff_s(z) - 1.0) / np.sqrt(stumpff_c(z))

    def time_error(z: float) -> float:
        yz = y(z)
        c, s = stumpff_c(z), stumpff_s(z)
        return (yz / c) ** 1.5 * s + a_const * np.sqrt(yz) - root_mu * tof_s

    high = _Z_MAX * (1.0 - 1e-6)
    low = -_Z_MAX
    if y(low) <= 0.0:
        # y rises with z; start just above its root so the square roots exist.
        lo, hi = low, high
        for _ in range(_BISECTION_ITERATIONS):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if y(mid) <= 0.0 else (lo, mid)
        low = hi
    while time_error(low) > 0.0:
        low = 2.0 * low - 1.0
        if low < -1e6 or y(low) <= 0.0:
            raise ValueError("no single-revolution transfer for this time of flight")
    if time_error(high) < 0.0:
        raise ValueError(
            "time of flight too long for a single revolution; "
            "multi-revolution transfers are not supported"
        )
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        if time_error(mid) < 0.0:
            low = mid
        else:
            high = mid
    z = 0.5 * (low + high)

    yz = y(z)
    f = 1.0 - yz / r1n
    g = a_const * np.sqrt(yz / mu_km3_s2)
    gdot = 1.0 - yz / r2n
    return (r2 - f * r1) / g, (gdot * r2 - r1) / g
