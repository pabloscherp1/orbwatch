"""Monte Carlo of the proximity operations under navigation and execution errors.

Each run flies the designed sequence with closed-loop guidance at every burn:
the burn is recomputed from a noisy estimate of the relative state, then
executed with errors in magnitude and pointing, and the true state is coasted
exactly with the Clohessy-Wiltshire solution. The outputs are the statistics
the budget and the safety case need: the speed change actually spent, the
closest approach to the target, and how often the keep-out sphere is entered.

Navigation errors are modelled as independent Gaussian noise on the estimate at
each burn, the way a camera and a rangefinder err: small across the line of
sight, set by the bearing accuracy, and larger along it, set by the range
accuracy. The distinction matters. Radial knowledge errors turn into in-track
drift, amplified some 35 times over most of an orbit, so an isotropic 1% of
range error made the transfers miss the safety ellipse by hundreds of metres.
No filter is simulated. Runs are vectorised, so a thousand cost well under a
second.

Units: m, m/s, s, degrees where named.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
from numpy.typing import NDArray

from orbwatch.rpo import cw
from orbwatch.rpo.approach import ProximityDesign


@dataclass(frozen=True)
class Dispersions:
    """One-sigma errors.

    Attributes
    ----------
    arrival_position_m, arrival_velocity_m_s : float
        Per axis, at the hold point where the far-range transfer ends.
    range_fraction : float
        Position knowledge along the line of sight, as a fraction of range.
    bearing_deg : float
        Direction knowledge, which sets position knowledge across the line of
        sight: range times this angle.
    navigation_floor_m : float
        Best position knowledge per axis, at short range.
    navigation_velocity_m_s : float
        Relative velocity knowledge per axis.
    magnitude_fraction : float
        Burn magnitude error, as a fraction of the commanded burn.
    pointing_deg : float
        Burn direction error.
    """

    arrival_position_m: float = 30.0
    arrival_velocity_m_s: float = 0.005
    range_fraction: float = 0.01
    bearing_deg: float = 0.05
    navigation_floor_m: float = 0.5
    navigation_velocity_m_s: float = 0.001
    magnitude_fraction: float = 0.02
    pointing_deg: float = 1.0

    def scaled(self, factor: float) -> Dispersions:
        """Every error multiplied by ``factor``: 2 is twice as bad throughout."""
        return replace(
            self, **{f.name: getattr(self, f.name) * factor for f in fields(self)}
        )


@dataclass(frozen=True)
class MonteCarloResult:
    runs: int
    dv_m_s: NDArray[np.float64]
    min_distance_m: NDArray[np.float64]
    keep_out_m: float
    nominal_dv_m_s: float
    burn_dv_m_s: NDArray[np.float64]
    """Speed change per run and per burn, shape (runs, burns)."""
    sample_times_s: NDArray[np.float64]
    sample_states: NDArray[np.float64]
    """A few runs' trajectories, shape (runs shown, samples, 6), for plotting."""

    def dv_percentile(self, q: float) -> float:
        return float(np.percentile(self.dv_m_s, q))

    @property
    def violation_fraction(self) -> float:
        return float(np.mean(self.min_distance_m < self.keep_out_m))


def _estimate(
    truth: NDArray, dispersions: Dispersions, rng: np.random.Generator
) -> NDArray:
    """Noisy estimate: range error along the line of sight, bearing error across."""
    runs = truth.shape[0]
    r = truth[:, :3]
    distance = np.linalg.norm(r, axis=1, keepdims=True)
    los = r / np.maximum(distance, 1e-9)
    across = rng.normal(size=(runs, 3))
    across -= np.sum(across * los, axis=1, keepdims=True) * los
    along_sigma = np.maximum(
        dispersions.navigation_floor_m, dispersions.range_fraction * distance
    )
    across_sigma = np.maximum(
        dispersions.navigation_floor_m, np.deg2rad(dispersions.bearing_deg) * distance
    )
    noise_r = los * rng.normal(size=(runs, 1)) * along_sigma + across * across_sigma
    noise_v = rng.normal(size=(runs, 3)) * dispersions.navigation_velocity_m_s
    return truth + np.concatenate([noise_r, noise_v], axis=1)


def _execute(
    commanded: NDArray, dispersions: Dispersions, rng: np.random.Generator
) -> NDArray:
    """Commanded burns with magnitude and pointing errors, Rodrigues rotation
    about a random axis perpendicular to each burn."""
    runs = commanded.shape[0]
    scale = 1.0 + rng.normal(size=(runs, 1)) * dispersions.magnitude_fraction
    size = np.linalg.norm(commanded, axis=1, keepdims=True)
    direction = np.divide(commanded, size, out=np.zeros_like(commanded), where=size > 0)
    axis = np.cross(direction, rng.normal(size=(runs, 3)))
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = np.divide(axis, norm, out=np.zeros_like(axis), where=norm > 0)
    angle = np.deg2rad(rng.normal(size=(runs, 1)) * dispersions.pointing_deg)
    rotated = (
        commanded * np.cos(angle)
        + np.cross(axis, commanded) * np.sin(angle)
        + axis * np.sum(axis * commanded, axis=1, keepdims=True) * (1 - np.cos(angle))
    )
    return rotated * scale


def _command(burn, estimate: NDArray, n: float) -> NDArray:
    """What guidance commands for one burn, given the estimated states."""
    r, v = estimate[:, :3], estimate[:, 3:]
    drift_free = -2.0 * n * r[:, 0]
    if burn.kind == "brake":
        wanted = np.zeros_like(v)
        wanted[:, 1] = drift_free
        return wanted - v
    if burn.kind == "hop":
        required = np.zeros_like(v)
        required[:, 0] = -n * (burn.target[1] - r[:, 1]) / 4.0
        required[:, 1] = drift_free
        return required - v
    if burn.kind in ("insert", "keep"):
        wanted = np.broadcast_to(burn.target[3:], v.shape).copy()
        wanted[:, 1] = -2.0 * n * r[:, 0]
        return wanted - v
    phi = cw.stm(n, burn.tof_s)
    prr, prv = phi[:3, :3], phi[:3, 3:]
    needed = np.linalg.solve(prv, (burn.target - r @ prr.T).T).T
    return needed - v


def run_monte_carlo(
    design: ProximityDesign,
    dispersions: Dispersions | None = None,
    runs: int = 500,
    seed: int = 1,
    shown_runs: int = 25,
    step_s: float = 60.0,
) -> MonteCarloResult:
    """Fly the design ``runs`` times with dispersions and closed-loop guidance.

    Guidance at each burn: a hop re-aims at its hold from the estimated
    in-track position; a brake cancels the estimated velocity; a transfer
    re-solves the two-impulse problem from the estimated state to its aim
    point; an insertion or maintenance burn puts the spacecraft on a drift-free
    ellipse through its estimated position. Brakes and hops also set the
    in-track speed to -2 n x, so a radial offset does not become drift.
    """
    dispersions = dispersions or Dispersions()
    rng = np.random.default_rng(seed)
    n = design.n
    first = design.burns[0].state_before
    truth = np.tile(first, (runs, 1))
    truth[:, :3] += rng.normal(size=(runs, 3)) * dispersions.arrival_position_m
    truth[:, 3:] += rng.normal(size=(runs, 3)) * dispersions.arrival_velocity_m_s

    spent = np.zeros((runs, len(design.burns)))
    closest = np.linalg.norm(truth[:, :3], axis=1)
    times, samples = [], []
    edges = [b.time_s for b in design.burns] + [design.end_time_s]
    for k, burn in enumerate(design.burns):
        commanded = _command(burn, _estimate(truth, dispersions, rng), n)
        executed = _execute(commanded, dispersions, rng)
        truth = truth.copy()
        truth[:, 3:] += executed
        spent[:, k] = np.linalg.norm(executed, axis=1)
        duration = edges[k + 1] - burn.time_s
        local = np.append(np.arange(0.0, duration, step_s), duration)
        coast = cw.propagate(truth, n, local)
        closest = np.minimum(
            closest, np.linalg.norm(coast[..., :3], axis=2).min(axis=0)
        )
        times.append(burn.time_s + local[:-1])
        samples.append(coast[:-1, :shown_runs])
        truth = coast[-1]

    return MonteCarloResult(
        runs=runs,
        dv_m_s=spent.sum(axis=1),
        min_distance_m=closest,
        keep_out_m=design.keep_out_m,
        nominal_dv_m_s=design.dv_m_s,
        burn_dv_m_s=spent,
        sample_times_s=np.concatenate(times),
        sample_states=np.concatenate(samples).transpose(1, 0, 2),
    )
