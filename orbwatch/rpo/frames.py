"""The target's radial, in-track, cross-track frame (RIC, or Hill frame).

Axes: radial x along the target's position, cross-track z along its angular
momentum, in-track y completing the right-handed set, close to the velocity
for a near-circular orbit. The frame rotates with the target, so a relative
velocity seen in it removes the frame's own rotation.

Inertial states are in km and km/s, as elsewhere in ORBWATCH; relative states
are in m and m/s, the scale of proximity operations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ric_basis(r_km: ArrayLike, v_km_s: ArrayLike) -> NDArray[np.float64]:
    """Rotation whose rows are the radial, in-track and cross-track unit vectors."""
    r = np.asarray(r_km, dtype=float)
    h = np.cross(r, np.asarray(v_km_s, dtype=float))
    x = r / np.linalg.norm(r)
    z = h / np.linalg.norm(h)
    return np.array([x, np.cross(z, x), z])


def _frame_rate(r_km: NDArray, v_km_s: NDArray) -> NDArray[np.float64]:
    """Angular velocity of the frame, expressed in the frame, rad/s."""
    h = np.linalg.norm(np.cross(r_km, v_km_s))
    return np.array([0.0, 0.0, h / float(np.dot(r_km, r_km))])


def inertial_to_ric(
    target_r_km: ArrayLike,
    target_v_km_s: ArrayLike,
    chaser_r_km: ArrayLike,
    chaser_v_km_s: ArrayLike,
) -> NDArray[np.float64]:
    """Chaser state relative to the target in its RIC frame, [m, m/s]."""
    rt, vt = np.asarray(target_r_km, float), np.asarray(target_v_km_s, float)
    basis = ric_basis(rt, vt)
    rho = basis @ (np.asarray(chaser_r_km, float) - rt)
    rho_dot = basis @ (np.asarray(chaser_v_km_s, float) - vt) - np.cross(
        _frame_rate(rt, vt), rho
    )
    return np.concatenate([rho, rho_dot]) * 1000.0


def ric_to_inertial(
    target_r_km: ArrayLike, target_v_km_s: ArrayLike, relative: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inertial chaser position and velocity, km and km/s, from a RIC state."""
    rt, vt = np.asarray(target_r_km, float), np.asarray(target_v_km_s, float)
    rel = np.asarray(relative, float) / 1000.0
    rho, rho_dot = rel[:3], rel[3:]
    basis = ric_basis(rt, vt)
    v_rel = rho_dot + np.cross(_frame_rate(rt, vt), rho)
    return rt + basis.T @ rho, vt + basis.T @ v_rel


def _unit(theta: float, phi: float) -> tuple[NDArray, NDArray, NDArray]:
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    u = np.array([cp * ct, cp * st, sp])
    du_dtheta = np.array([-cp * st, cp * ct, 0.0])
    du_dphi = np.array([-sp * ct, -sp * st, cp])
    return u, du_dtheta, du_dphi


def inertial_to_curvilinear(
    target_r_km: ArrayLike,
    target_v_km_s: ArrayLike,
    chaser_r_km: ArrayLike,
    chaser_v_km_s: ArrayLike,
) -> NDArray[np.float64]:
    """Chaser state in curvilinear RIC coordinates, [m, m/s].

    Radial is the difference in radius; in-track and cross-track are arc
    lengths at the target's radius. Clohessy-Wiltshire holds in these to first
    order just as in straight-line coordinates, but a point several kilometres
    behind on the target's own orbit is at zero radial offset, as it should
    be. In straight-line coordinates it would sit y^2 / 2a above the orbit, and
    at 5 km in low orbit that 1.75 m makes the model drift 55 m per orbit.
    """
    rt, vt = np.asarray(target_r_km, float), np.asarray(target_v_km_s, float)
    basis = ric_basis(rt, vt)
    omega = _frame_rate(rt, vt)
    p = basis @ np.asarray(chaser_r_km, float)
    p_dot = basis @ np.asarray(chaser_v_km_s, float) - np.cross(omega, p)
    r_t = float(np.linalg.norm(rt))
    r_t_dot = float(np.dot(rt, vt)) / r_t
    r = float(np.linalg.norm(p))
    r_dot = float(np.dot(p, p_dot)) / r
    theta = float(np.arctan2(p[1], p[0]))
    phi = float(np.arcsin(p[2] / r))
    theta_dot = (p[0] * p_dot[1] - p[1] * p_dot[0]) / (p[0] ** 2 + p[1] ** 2)
    phi_dot = (p_dot[2] * r - p[2] * r_dot) / (r * r * np.cos(phi))
    state = np.array(
        [
            r - r_t,
            r_t * theta,
            r_t * phi,
            r_dot - r_t_dot,
            r_t * theta_dot + r_t_dot * theta,
            r_t * phi_dot + r_t_dot * phi,
        ]
    )
    return state * 1000.0


def curvilinear_to_inertial(
    target_r_km: ArrayLike, target_v_km_s: ArrayLike, relative: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inertial chaser position and velocity, km and km/s, from curvilinear RIC."""
    rt, vt = np.asarray(target_r_km, float), np.asarray(target_v_km_s, float)
    x, y, z, x_dot, y_dot, z_dot = np.asarray(relative, float) / 1000.0
    basis = ric_basis(rt, vt)
    omega = _frame_rate(rt, vt)
    r_t = float(np.linalg.norm(rt))
    r_t_dot = float(np.dot(rt, vt)) / r_t
    theta, phi = y / r_t, z / r_t
    theta_dot = (y_dot - r_t_dot * theta) / r_t
    phi_dot = (z_dot - r_t_dot * phi) / r_t
    r, r_dot = r_t + x, r_t_dot + x_dot
    u, du_dtheta, du_dphi = _unit(theta, phi)
    p = r * u
    p_dot = r_dot * u + r * (du_dtheta * theta_dot + du_dphi * phi_dot)
    return basis.T @ p, basis.T @ (p_dot + np.cross(omega, p))
