"""Delta-v budgets, and whether the detected manoeuvres account for them.

A detector can only report what it sees. The budget is the check on what it did
not see: work out, from physics alone, how much speed change the orbit's history
requires, and compare the detected total against it. The ratio is the closure.

- **Low orbit, in-plane.** The semi-major axis ends the span somewhere, drag
  removed a known amount on the way, so reboosts must have added the
  difference. Validated on the ISS: detected reboosts closed the year to 101.5%.
- **Geostationary, out-of-plane.** A satellite held near zero inclination must
  undo the natural precession of its orbit plane, whether it burns in visible
  steps or near-continuously. This is the containment argument, and it is the
  only way to measure satellites whose individual corrections sit below the
  noise.
- **Geostationary, in-plane.** East-west keeping corrects in both directions, so
  the net change bounds nothing and no requirement is computed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from orbwatch.events.detect import Kind, Manoeuvre, StepAnalysis
from orbwatch.events.history import OrbitHistory

LAPLACE_POLE_INCLINATION_RAD: float = float(np.deg2rad(7.4))
"""Inclination of the Laplace plane at geostationary radius, about 7.4 deg.

Established in the literature on long-term GEO dynamics; confirmed through search
summaries rather than the primary papers.
"""

LAPLACE_PRECESSION_PERIOD_DAYS: float = 53.0 * 365.25
"""Period of the inclination vector's precession about the Laplace pole, ~53 yr."""

ENDPOINT_WINDOW_DAYS: float = 2.0


def natural_inclination_rate_rad_per_day(
    inclination_vector_rad: ArrayLike,
) -> NDArray[np.float64]:
    """Natural drift of the inclination vector of an uncontrolled GEO object.

    Parameters
    ----------
    inclination_vector_rad : array_like, shape (2,) or (N, 2)
        (i sin RAAN, i cos RAAN), radians.

    Returns
    -------
    ndarray, same shape
        Rate of change, radians per day.

    Notes
    -----
    A simplified model: the vector rotates counter-clockwise about the Laplace
    pole at (0, 7.4 deg) once every 53 years. The sense of rotation was checked
    against INTELSAT 905's measured motion, and the model's rate came within 9%
    of that satellite's measured 1.04 deg per year. It ignores the Sun's
    six-monthly and the Moon's fortnightly terms, which average out over a year
    but not over days.

    At zero inclination the push points toward a node of 90 deg at about
    0.88 deg per year, roughly 47 m/s per year of north-south station-keeping.
    """
    v = np.asarray(inclination_vector_rad, dtype=float)
    relative = v - np.array([0.0, LAPLACE_POLE_INCLINATION_RAD])
    omega = 2.0 * np.pi / LAPLACE_PRECESSION_PERIOD_DAYS
    return omega * np.stack([-relative[..., 1], relative[..., 0]], axis=-1)


@dataclass(frozen=True)
class Budget:
    """Detected and required speed change for one signal over a history.

    ``required_m_s`` is None where physics gives no requirement, as for
    east-west keeping. ``closure`` is detected over required.
    """

    kind: Kind
    span_days: float
    detected_m_s: float
    required_m_s: float | None
    method: str

    @property
    def closure(self) -> float | None:
        if self.required_m_s is None or self.required_m_s <= 0:
            return None
        return self.detected_m_s / self.required_m_s

    def per_year(self, value_m_s: float | None) -> float | None:
        if value_m_s is None or self.span_days <= 0:
            return None
        return value_m_s * 365.25 / self.span_days


def _endpoint(t: NDArray, x: NDArray, first: bool) -> NDArray:
    """Robust state at the start or end: median of the first or last few days."""
    mask = (
        t <= t[0] + ENDPOINT_WINDOW_DAYS if first else t >= t[-1] - ENDPOINT_WINDOW_DAYS
    )
    return np.median(x[mask], axis=0)


def in_plane_budget(
    history: OrbitHistory, analysis: StepAnalysis, manoeuvres: list[Manoeuvre]
) -> Budget:
    """In-plane budget. In low orbit, checks detected raises against drag loss.

    The requirement is the altitude manoeuvres must have added: the net change
    in semi-major axis minus the natural decay. Only raising manoeuvres count
    against it. A lowering manoeuvre is still reported as a manoeuvre, and a
    sustained drop from a drag surge is not propellant at all.
    """
    in_plane = [m for m in manoeuvres if m.kind == "in_plane"]
    if history.regime == "geosynchronous":
        detected = sum(m.delta_v_m_s for m in in_plane)
        return Budget(
            "in_plane",
            history.span_days,
            detected,
            None,
            "east-west keeping corrects in both directions; no requirement",
        )
    detected = sum(m.delta_v_m_s for m in in_plane if (m.delta_a_km or 0.0) > 0.0)
    natural_km = float((analysis.local_rate[:, 0] * analysis.dt_days).sum())
    net_km = float(history.a_km[-1] - history.a_km[0])
    added_km = net_km - natural_km
    v_m_s = history.mean_speed_km_s * 1000.0
    required = 0.5 * v_m_s * max(added_km, 0.0) / float(np.mean(history.a_km))
    return Budget(
        "in_plane",
        history.span_days,
        detected,
        required,
        "raises detected against net change in semi-major axis minus natural decay",
    )


def out_of_plane_budget(
    history: OrbitHistory, analysis: StepAnalysis, manoeuvres: list[Manoeuvre]
) -> Budget:
    detected = sum(m.delta_v_m_s for m in manoeuvres if m.kind == "out_of_plane")
    if history.regime != "geosynchronous":
        return Budget(
            "out_of_plane",
            history.span_days,
            detected,
            None,
            "no natural out-of-plane model outside GEO",
        )
    t, ivec = history.t_days, history.inclination_vector_rad
    dt = np.diff(t)
    midpoints = 0.5 * (ivec[1:] + ivec[:-1])
    natural = (natural_inclination_rate_rad_per_day(midpoints) * dt[:, None]).sum(
        axis=0
    )
    observed = _endpoint(t, ivec, first=False) - _endpoint(t, ivec, first=True)
    control = observed - natural
    required = history.mean_speed_km_s * 1000.0 * float(np.linalg.norm(control))
    return Budget(
        "out_of_plane",
        history.span_days,
        detected,
        required,
        "containment against the Laplace-plane precession model (lower bound)",
    )
