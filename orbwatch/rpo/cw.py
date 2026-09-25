"""Clohessy-Wiltshire relative motion about a target in a circular orbit.

Close to a target, the difference between two orbits obeys linear equations in
the target's rotating radial, in-track, cross-track frame (the Hill frame):

    x'' - 3 n^2 x - 2 n y' = 0      radial, positive away from Earth
    y'' + 2 n x'           = 0      in-track, positive along the velocity
    z'' + n^2 z            = 0      cross-track, along the angular momentum

with n the target's mean motion. They have an exact solution, the state
transition matrix below, which makes guidance and Monte Carlo cheap: a state at
any later time is one matrix product away. The linearisation drops terms of
order separation squared over orbit radius, under 2 m at 5 km in low orbit.

Units: metres, metres per second, seconds, radians.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_SINGULAR_CONDITION: float = 1e8


def stm(n: float, t: float | ArrayLike) -> NDArray[np.float64]:
    """Clohessy-Wiltshire state transition matrix.

    Parameters
    ----------
    n : float
        Target mean motion, rad/s.
    t : float or array_like
        Elapsed time, s.

    Returns
    -------
    ndarray, shape (6, 6) or (..., 6, 6)
        Maps [x, y, z, x', y', z'] at 0 to the state at ``t``.
    """
    t = np.asarray(t, dtype=float)
    s, c = np.sin(n * t), np.cos(n * t)
    nt = n * t
    phi = np.zeros(t.shape + (6, 6))
    phi[..., 0, 0] = 4.0 - 3.0 * c
    phi[..., 1, 0] = 6.0 * (s - nt)
    phi[..., 1, 1] = 1.0
    phi[..., 2, 2] = c
    phi[..., 0, 3] = s / n
    phi[..., 0, 4] = 2.0 * (1.0 - c) / n
    phi[..., 1, 3] = -2.0 * (1.0 - c) / n
    phi[..., 1, 4] = (4.0 * s - 3.0 * nt) / n
    phi[..., 2, 5] = s / n
    phi[..., 3, 0] = 3.0 * n * s
    phi[..., 4, 0] = -6.0 * n * (1.0 - c)
    phi[..., 5, 2] = -n * s
    phi[..., 3, 3] = c
    phi[..., 3, 4] = 2.0 * s
    phi[..., 4, 3] = -2.0 * s
    phi[..., 4, 4] = 4.0 * c - 3.0
    phi[..., 5, 5] = c
    return phi


def propagate(state: ArrayLike, n: float, t: float | ArrayLike) -> NDArray[np.float64]:
    """Relative state after ``t`` seconds of free motion.

    ``state`` may be one state, shape (6,), or many, shape (..., 6), and ``t``
    one time or many. The result has shape t.shape + state.shape: times first,
    then states.
    """
    x = np.asarray(state, dtype=float)
    phi = stm(n, t)
    out = np.einsum("tij,sj->tsi", phi.reshape(-1, 6, 6), x.reshape(-1, 6))
    return out.reshape(np.shape(t) + x.shape)


def two_impulse(
    position0: ArrayLike,
    velocity0: ArrayLike,
    position1: ArrayLike,
    n: float,
    tof: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """First burn of the transfer to ``position1`` in ``tof``, and the arrival velocity.

    Solves r(tof) = Prr r0 + Prv v0 for the departure velocity. Transfers of
    a whole number of half orbits are singular in radial and cross-track and
    are refused rather than solved badly.

    Returns
    -------
    dv1 : ndarray, shape (3,)
        Departure burn, m/s.
    arrival_velocity : ndarray, shape (3,)
        Velocity on reaching ``position1``, before any second burn.

    Raises
    ------
    ValueError
        For a singular time of flight.
    """
    phi = stm(n, tof)
    prr, prv = phi[:3, :3], phi[:3, 3:]
    vrr, vrv = phi[3:, :3], phi[3:, 3:]
    if np.linalg.cond(prv) > _SINGULAR_CONDITION:
        raise ValueError(
            "time of flight is a whole number of half orbits: the transfer is singular"
        )
    r0 = np.asarray(position0, dtype=float)
    v_needed = np.linalg.solve(prv, np.asarray(position1, dtype=float) - prr @ r0)
    return v_needed - np.asarray(velocity0, dtype=float), vrr @ r0 + vrv @ v_needed


def radial_hop_speed(n: float, in_track_change: float) -> float:
    """Radial speed that moves a spacecraft along the V-bar in half an orbit.

    From rest on the in-track axis, a radial burn traces a closed ellipse that
    crosses the axis again half an orbit later, displaced by -4 x'/n. So a hop
    of ``in_track_change`` needs x' = -n dy / 4, and the same again to stop.
    If the second burn fails the ellipse brings the spacecraft back to its
    start, which is what makes radial hops passively safe.
    """
    return -n * in_track_change / 4.0


def safety_ellipse_state(
    radial_m: float, cross_track_m: float, n: float, phase_rad: float
) -> NDArray[np.float64]:
    """State on a passively safe, drift-free relative orbit about the target.

    x = A sin(psi), y = 2A cos(psi), z = B cos(psi): an ellipse twice as long
    in-track as radially, centred on the target, with the cross-track motion a
    quarter orbit out of phase with the radial one. Whenever the radial
    separation passes through zero the cross-track separation is at its
    largest, so the path never crosses the target's in-track axis. If an error
    makes the ellipse drift in-track, the spacecraft still passes the target no
    closer than min(A, B). This is eccentricity and inclination vector
    separation, the passive-safety principle of formation flying.
    """
    a, b = radial_m, cross_track_m
    s, c = np.sin(phase_rad), np.cos(phase_rad)
    return np.array(
        [a * s, 2.0 * a * c, b * c, a * n * c, -2.0 * a * n * s, -b * n * s]
    )


def in_track_drift_per_orbit(state: ArrayLike, n: float) -> float:
    """Net in-track motion per orbit, m: -6 pi (y' + 2 n x) / n.

    The secular term of the in-track solution is -(6 n x + 3 y') t. It
    vanishes exactly when y' = -2 n x, the drift-free condition.
    """
    x, _, _, _, ydot, _ = np.asarray(state, dtype=float)
    return float(-6.0 * np.pi * (ydot + 2.0 * n * x) / n)
