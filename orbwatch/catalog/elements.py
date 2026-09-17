"""Classical orbital elements from and to inertial Cartesian state.

Everything here works in an Earth-centred inertial frame. No frame conversion
happens in this module. Distances are km, speeds are km/s, angles are radians,
and the gravitational parameter is km^3/s^2.

Both directions are built from the same three orthonormal basis vectors, so the
forward and inverse conversions read as mirror images of each other:

- ``node_hat``, along the line of nodes, where the orbit crosses the equator
  going north.
- ``in_plane_hat``, in the orbit plane, ninety degrees ahead of the node in the
  direction of motion.
- ``h_hat``, the orbit plane normal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

MU_EARTH_KM3_S2: float = 398600.4418
"""Earth gravitational parameter, km^3/s^2. EGM-96 value, textbook constant."""

TWO_PI: float = 2.0 * np.pi

CIRCULAR_ECCENTRICITY_TOL: float = 1e-8
"""Below this eccentricity the orbit is treated as circular.

Dimensionless by construction. At geostationary radius, e = 1e-8 is a radial
variation of about 40 cm over a revolution, far below any orbit determination
accuracy that will ever be fed to this module.
"""

EQUATORIAL_SINE_INCLINATION_TOL: float = 1e-8
"""Below this value of sin(i) the orbit is treated as equatorial.

The node vector has magnitude |z_hat x h| = h sin(i), so dividing by h gives a
dimensionless test that does not care about the size of the orbit.
"""

PARABOLIC_ENERGY_TOL: float = 1e-12
"""Specific energy, km^2/s^2, below which the semi-major axis is meaningless."""


def _wrap_to_two_pi(angle_rad: float) -> float:
    """Wrap an angle into [0, 2*pi)."""
    return float(angle_rad % TWO_PI)


def _perifocal_basis(
    i_rad: float, raan_rad: float, argp_rad: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the perifocal basis expressed in the inertial frame.

    Returns ``(periapsis_hat, semi_latus_hat, h_hat)``, a right-handed triad
    where ``periapsis_hat`` points at periapsis and ``semi_latus_hat`` is ninety
    degrees ahead of it in the direction of motion.
    """
    cos_raan, sin_raan = np.cos(raan_rad), np.sin(raan_rad)
    cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)
    cos_argp, sin_argp = np.cos(argp_rad), np.sin(argp_rad)

    node_hat = np.array([cos_raan, sin_raan, 0.0])
    h_hat = np.array([sin_raan * sin_i, -cos_raan * sin_i, cos_i])
    in_plane_hat = np.cross(h_hat, node_hat)

    periapsis_hat = cos_argp * node_hat + sin_argp * in_plane_hat
    semi_latus_hat = -sin_argp * node_hat + cos_argp * in_plane_hat
    return periapsis_hat, semi_latus_hat, h_hat


@dataclass(frozen=True)
class OrbitalElements:
    """A classical element set, with the gravitational parameter it assumes.

    Immutable on purpose. Carrying ``mu_km3_s2`` alongside the elements closes a
    real trap: an element set derived under one gravitational parameter being
    reused under another, which produces a quietly wrong period and a quietly
    wrong delta-v downstream.

    Attributes
    ----------
    a_km : float
        Semi-major axis, km.
    e : float
        Eccentricity, unitless.
    i_rad : float
        Inclination, rad, in [0, pi].
    raan_rad : float
        Right ascension of the ascending node, rad, in [0, 2*pi).
    argp_rad : float
        Argument of periapsis, rad, in [0, 2*pi).
    nu_rad : float
        True anomaly, rad, in [0, 2*pi).
    mu_km3_s2 : float
        Gravitational parameter these elements were derived under, km^3/s^2.
    """

    a_km: float
    e: float
    i_rad: float
    raan_rad: float
    argp_rad: float
    nu_rad: float
    mu_km3_s2: float = MU_EARTH_KM3_S2

    @property
    def semi_latus_rectum_km(self) -> float:
        """Semi-latus rectum p = a(1 - e^2), km."""
        return self.a_km * (1.0 - self.e**2)

    @property
    def r_periapsis_km(self) -> float:
        return self.a_km * (1.0 - self.e)

    @property
    def r_apoapsis_km(self) -> float:
        return self.a_km * (1.0 + self.e)

    @property
    def mean_motion_rad_s(self) -> float:
        """Mean motion n = sqrt(mu / a^3), rad/s. Also the Hill frame rate."""
        return float(np.sqrt(self.mu_km3_s2 / self.a_km**3))

    @property
    def period_s(self) -> float:
        return TWO_PI / self.mean_motion_rad_s

    @property
    def argument_of_latitude_rad(self) -> float:
        """Angle from the ascending node to the spacecraft, rad.

        The meaningful in-plane angle for a circular orbit, where the argument
        of periapsis and the true anomaly are individually undefined.
        """
        return _wrap_to_two_pi(self.argp_rad + self.nu_rad)


def state_to_elements(
    r_eci_km: ArrayLike,
    v_eci_km_s: ArrayLike,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
) -> OrbitalElements:
    """Convert an inertial Cartesian state to classical orbital elements.

    Parameters
    ----------
    r_eci_km : array_like, shape (3,)
        Position in an Earth-centred inertial frame, km.
    v_eci_km_s : array_like, shape (3,)
        Velocity in the same Earth-centred inertial frame, km/s.
    mu_km3_s2 : float, optional
        Gravitational parameter of the central body, km^3/s^2.

    Returns
    -------
    OrbitalElements

    Raises
    ------
    ValueError
        If the state is near-parabolic, where the semi-major axis diverges, or
        if the position or velocity is degenerate.

    Notes
    -----
    Two-body motion only. Nothing here knows about J2, drag, solar radiation
    pressure or third bodies.

    Angles come out of ``arctan2`` on in-plane basis projections rather than out
    of ``arccos`` with quadrant tests. ``arccos`` loses about half the available
    significant digits near its endpoints because its derivative diverges there,
    so recovering an angle of zero through it is accurate to only about 1e-8
    rad. ``arctan2`` reaches machine precision and resolves the quadrant free.

    Degenerate cases, both coordinate artefacts rather than physics, and both of
    which a near-circular near-equatorial GEO object sits on simultaneously:

    - Circular, ``e`` below ``CIRCULAR_ECCENTRICITY_TOL``: periapsis does not
      exist, so the argument of periapsis is undefined. Returns
      ``argp_rad = 0`` and puts the whole in-plane angle into ``nu_rad``, which
      then equals the argument of latitude.
    - Equatorial, ``sin(i)`` below ``EQUATORIAL_SINE_INCLINATION_TOL``: the line
      of nodes does not exist, so the right ascension of the ascending node is
      undefined. Returns ``raan_rad = 0`` and measures in-plane angles from the
      inertial x axis.

    In both cases ``argument_of_latitude_rad`` remains meaningful. Prefer
    equinoctial elements anywhere the singular behaviour would propagate, such
    as covariance handling or numerical integration.
    """
    r_vec_km: NDArray[np.float64] = np.asarray(r_eci_km, dtype=float)
    v_vec_km_s: NDArray[np.float64] = np.asarray(v_eci_km_s, dtype=float)

    if r_vec_km.shape != (3,) or v_vec_km_s.shape != (3,):
        raise ValueError("position and velocity must each have shape (3,)")

    r_km = float(np.linalg.norm(r_vec_km))
    v_km_s = float(np.linalg.norm(v_vec_km_s))
    if r_km == 0.0 or v_km_s == 0.0:
        raise ValueError("position and velocity must both be non-zero")

    # Specific angular momentum. Constant under two-body motion, and its
    # direction is the orbit plane normal.
    h_vec_km2_s = np.cross(r_vec_km, v_vec_km_s)
    h_km2_s = float(np.linalg.norm(h_vec_km2_s))
    if h_km2_s == 0.0:
        raise ValueError("position and velocity are parallel, orbit is rectilinear")

    # Specific energy, then vis-viva for the semi-major axis.
    energy_km2_s2 = 0.5 * v_km_s**2 - mu_km3_s2 / r_km
    if abs(energy_km2_s2) < PARABOLIC_ENERGY_TOL:
        raise ValueError("state is near-parabolic, semi-major axis diverges")
    a_km = -mu_km3_s2 / (2.0 * energy_km2_s2)

    # Eccentricity vector. Lies in the orbit plane and points at periapsis.
    e_vec = (
        (v_km_s**2 - mu_km3_s2 / r_km) * r_vec_km
        - float(np.dot(r_vec_km, v_vec_km_s)) * v_vec_km_s
    ) / mu_km3_s2
    e = float(np.linalg.norm(e_vec))

    h_hat = h_vec_km2_s / h_km2_s
    i_rad = float(np.arccos(np.clip(h_hat[2], -1.0, 1.0)))

    # Node vector, along the line of nodes. cross(z_hat, h) reduces to
    # (-h_y, h_x, 0), and its magnitude is h sin(i).
    n_vec = np.array([-h_vec_km2_s[1], h_vec_km2_s[0], 0.0])
    sine_inclination = float(np.linalg.norm(n_vec)) / h_km2_s

    if sine_inclination < EQUATORIAL_SINE_INCLINATION_TOL:
        raan_rad = 0.0
        node_hat = np.array([1.0, 0.0, 0.0])
    else:
        raan_rad = float(np.arctan2(n_vec[1], n_vec[0]))
        node_hat = n_vec / float(np.linalg.norm(n_vec))

    # Complete a right-handed in-plane basis. {node_hat, in_plane_hat, h_hat} is
    # right-handed, and in_plane_hat points along the direction of motion at the
    # ascending node, so angles measured from node_hat towards in_plane_hat
    # increase in the direction of motion.
    in_plane_hat = np.cross(h_hat, node_hat)

    argument_of_latitude_rad = float(
        np.arctan2(np.dot(r_vec_km, in_plane_hat), np.dot(r_vec_km, node_hat))
    )

    if e < CIRCULAR_ECCENTRICITY_TOL:
        argp_rad = 0.0
        nu_rad = argument_of_latitude_rad
    else:
        argp_rad = float(
            np.arctan2(np.dot(e_vec, in_plane_hat), np.dot(e_vec, node_hat))
        )
        nu_rad = argument_of_latitude_rad - argp_rad

    return OrbitalElements(
        a_km=a_km,
        e=e,
        i_rad=i_rad,
        raan_rad=_wrap_to_two_pi(raan_rad),
        argp_rad=_wrap_to_two_pi(argp_rad),
        nu_rad=_wrap_to_two_pi(nu_rad),
        mu_km3_s2=mu_km3_s2,
    )


def elements_to_state(
    elements: OrbitalElements,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert classical orbital elements to an inertial Cartesian state.

    Parameters
    ----------
    elements : OrbitalElements
        Element set, including the gravitational parameter it assumes.

    Returns
    -------
    r_eci_km : ndarray, shape (3,)
        Position in an Earth-centred inertial frame, km.
    v_eci_km_s : ndarray, shape (3,)
        Velocity in the same frame, km/s.

    Raises
    ------
    ValueError
        If the element set is not a bound ellipse.

    Notes
    -----
    Elliptical orbits only, matching ``state_to_elements``, which refuses
    parabolic states. Extend both together if hyperbolic support is ever needed.

    In the perifocal frame the conic and the velocity are

        r = p / (1 + e cos nu),
        r_vec = r (cos nu, sin nu, 0),
        v_vec = sqrt(mu / p) (-sin nu, e + cos nu, 0),

    which is then projected onto the inertial frame with the perifocal basis.
    Because the degenerate conventions in ``state_to_elements`` fold the
    undefined angle into a defined one rather than discarding it, this function
    inverts those cases exactly and the round trip closes.
    """
    if not 0.0 <= elements.e < 1.0:
        raise ValueError(
            f"eccentricity {elements.e} is outside [0, 1), orbit is not a bound ellipse"
        )
    if elements.a_km <= 0.0:
        raise ValueError(f"semi-major axis {elements.a_km} km must be positive")

    p_km = elements.semi_latus_rectum_km
    cos_nu, sin_nu = np.cos(elements.nu_rad), np.sin(elements.nu_rad)
    r_km = p_km / (1.0 + elements.e * cos_nu)

    periapsis_hat, semi_latus_hat, _ = _perifocal_basis(
        elements.i_rad, elements.raan_rad, elements.argp_rad
    )

    r_eci_km = r_km * (cos_nu * periapsis_hat + sin_nu * semi_latus_hat)
    v_eci_km_s = np.sqrt(elements.mu_km3_s2 / p_km) * (
        -sin_nu * periapsis_hat + (elements.e + cos_nu) * semi_latus_hat
    )
    return r_eci_km, v_eci_km_s
