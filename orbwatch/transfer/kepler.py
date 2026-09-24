"""Two-body propagation with universal variables.

One formulation covers ellipses, parabolas and hyperbolas, so transfer arcs of
any shape can be propagated and checked without switching between anomalies.
Follows Curtis, Orbital Mechanics for Engineering Students, algorithms 3.3 and
3.4.

Units: km, km/s, seconds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from orbwatch.catalog.elements import MU_EARTH_KM3_S2

_SERIES_Z: float = 1e-6
_NEWTON_TOLERANCE: float = 1e-10
_NEWTON_MAX_ITERATIONS: int = 100


def stumpff_c(z: float) -> float:
    """Stumpff function C(z), with a series near zero where the closed form
    loses precision to cancellation."""
    if abs(z) < _SERIES_Z:
        return 0.5 - z / 24.0 + z * z / 720.0
    if z > 0.0:
        return (1.0 - np.cos(np.sqrt(z))) / z
    return (np.cosh(np.sqrt(-z)) - 1.0) / -z


def stumpff_s(z: float) -> float:
    """Stumpff function S(z), with a series near zero."""
    if abs(z) < _SERIES_Z:
        return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0
    if z > 0.0:
        root = np.sqrt(z)
        return (root - np.sin(root)) / root**3
    root = np.sqrt(-z)
    return (np.sinh(root) - root) / root**3


def propagate_kepler(
    r0_km: ArrayLike,
    v0_km_s: ArrayLike,
    dt_s: float,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Position and velocity after ``dt_s`` seconds of two-body motion.

    Solves the universal Kepler equation for the universal anomaly by Newton's
    method, then applies the Lagrange coefficients.

    Parameters
    ----------
    r0_km, v0_km_s : array_like, shape (3,)
        Initial state in any inertial frame.
    dt_s : float
        Time of flight, seconds; negative propagates backwards.
    mu_km3_s2 : float, optional

    Returns
    -------
    r_km, v_km_s : ndarray, shape (3,)
    """
    r0 = np.asarray(r0_km, dtype=float)
    v0 = np.asarray(v0_km_s, dtype=float)
    r0n = float(np.linalg.norm(r0))
    vr0 = float(np.dot(r0, v0)) / r0n
    alpha = 2.0 / r0n - float(np.dot(v0, v0)) / mu_km3_s2
    root_mu = np.sqrt(mu_km3_s2)

    chi = root_mu * abs(alpha) * dt_s
    if chi == 0.0:
        chi = root_mu * dt_s / r0n
    for _ in range(_NEWTON_MAX_ITERATIONS):
        z = alpha * chi * chi
        c, s = stumpff_c(z), stumpff_s(z)
        f = (
            r0n * vr0 / root_mu * chi * chi * c
            + (1.0 - alpha * r0n) * chi**3 * s
            + r0n * chi
            - root_mu * dt_s
        )
        df = (
            r0n * vr0 / root_mu * chi * (1.0 - alpha * chi * chi * s)
            + (1.0 - alpha * r0n) * chi * chi * c
            + r0n
        )
        step = f / df
        chi -= step
        if abs(step) < _NEWTON_TOLERANCE:
            break
    else:
        raise RuntimeError("universal Kepler equation did not converge")

    z = alpha * chi * chi
    c, s = stumpff_c(z), stumpff_s(z)
    f_lag = 1.0 - chi * chi / r0n * c
    g_lag = dt_s - chi**3 / root_mu * s
    r = f_lag * r0 + g_lag * v0
    rn = float(np.linalg.norm(r))
    fdot = root_mu / (rn * r0n) * (alpha * chi**3 * s - chi)
    gdot = 1.0 - chi * chi / rn * c
    return r, fdot * r0 + gdot * v0
