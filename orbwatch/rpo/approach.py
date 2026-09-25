"""Approach and inspection design, from the far-range hold point to a safety
ellipse around the target and back out.

The sequence, all in the target's curvilinear radial, in-track, cross-track
frame:

1. Hold on the V-bar behind the target, where the far-range transfer ended.
2. Radial hops along the V-bar to closer holds. Each hop is a closed relative
   orbit, so if its braking burn fails the spacecraft coasts back to where the
   hop started: passively safe by construction.
3. A two-impulse transfer from the last hold into a safety ellipse, choosing
   the time of flight that is cheapest among those where a failed insertion
   burn never brings the spacecraft inside the keep-out sphere. A mid-course
   correction halfway, zero in the nominal case, re-aims from a fresh and
   closer navigation fix; without it 0.4% of Monte Carlo runs entered the
   keep-out sphere on this leg.
4. Inspection on the safety ellipse, with one small maintenance burn per orbit
   planned (zero in the nominal case) to cancel drift from errors.
5. A two-impulse departure back to a hold on the V-bar, chosen the same way.

Every burn is then checked for the failure case: coast freely from the state
just before it for a set number of orbits and report the closest approach.

Units: m, m/s, s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orbwatch.rpo import cw

BurnKind = Literal["hop", "brake", "transfer", "insert", "keep"]

HOLD_M: float = 5000.0
WAYPOINTS_M: tuple[float, ...] = (1500.0,)
ELLIPSE_RADIAL_M: float = 250.0
ELLIPSE_CROSS_TRACK_M: float = 250.0
KEEP_OUT_M: float = 200.0
INSPECTION_ORBITS: int = 5
DEPARTURE_M: float = 1500.0
SAFETY_ORBITS: float = 2.0
TOF_FRACTIONS: tuple[float, ...] = tuple(
    float(f) for f in np.round(np.arange(0.10, 0.951, 0.025), 3) if abs(f - 0.5) > 0.04
)
"""Times of flight tried for the two-impulse transfers, as fractions of an
orbit. Half an orbit is excluded: radial and cross-track are singular there."""


@dataclass(frozen=True)
class Burn:
    """One planned impulse.

    ``target`` is what guidance aims at when the burn is recomputed from a
    noisy state: the waypoint position for a ``hop`` or ``transfer``, the
    safety-ellipse state for ``insert`` and ``keep``, nothing for ``brake``.
    """

    label: str
    kind: BurnKind
    time_s: float
    dv: NDArray[np.float64]
    state_before: NDArray[np.float64]
    tof_s: float = 0.0
    target: NDArray[np.float64] | None = None

    @property
    def dv_m_s(self) -> float:
        return float(np.linalg.norm(self.dv))

    @property
    def state_after(self) -> NDArray[np.float64]:
        after = self.state_before.copy()
        after[3:] += self.dv
        return after


@dataclass(frozen=True)
class SafetyCheck:
    """What happens if one burn fails: the closest approach while coasting.
    ``burn`` indexes the design's burns."""

    burn: int
    label: str
    min_distance_m: float
    time_after_s: float
    passes: bool


@dataclass(frozen=True)
class ProximityDesign:
    """The nominal approach, inspection and departure."""

    n: float
    burns: tuple[Burn, ...]
    end_time_s: float
    keep_out_m: float
    ellipse_radial_m: float
    ellipse_cross_track_m: float
    inspection_orbits: int
    safety_orbits: float

    @property
    def period_s(self) -> float:
        return 2.0 * np.pi / self.n

    @property
    def dv_m_s(self) -> float:
        return float(sum(b.dv_m_s for b in self.burns))

    @property
    def ellipse_separation_m(self) -> float:
        """Closest the ellipse passes to the target's in-track axis, in the
        radial and cross-track plane. It stays passively safe against in-track
        drift as long as this exceeds the keep-out radius."""
        return float(min(self.ellipse_radial_m, self.ellipse_cross_track_m))

    def trajectory(self, step_s: float = 60.0) -> tuple[NDArray, NDArray]:
        """Nominal times and states, sampled across every coast."""
        times, states = [], []
        edges = [b.time_s for b in self.burns] + [self.end_time_s]
        for burn, end in zip(self.burns, edges[1:], strict=True):
            local = np.arange(0.0, end - burn.time_s, step_s)
            times.append(burn.time_s + local)
            states.append(cw.propagate(burn.state_after, self.n, local))
        return np.concatenate(times), np.concatenate(states)


def _coast_min_distance(
    state: NDArray, n: float, duration_s: float, samples: int = 400
) -> tuple[float, float]:
    t = np.linspace(0.0, duration_s, samples)
    distance = np.linalg.norm(cw.propagate(state, n, t)[:, :3], axis=1)
    k = int(np.argmin(distance))
    return float(distance[k]), float(t[k])


def _choose_transfer(
    start: NDArray,
    goal_state: NDArray,
    n: float,
    keep_out_m: float,
    safety_orbits: float,
) -> tuple[NDArray, NDArray, float]:
    """Cheapest two-impulse transfer whose coast and whose failed arrival burn
    both stay outside the keep-out sphere; the cheapest overall if none does."""
    period = 2.0 * np.pi / n
    best, best_any = None, None
    for fraction in TOF_FRACTIONS:
        tof = fraction * period
        try:
            dv1, arrival = cw.two_impulse(start[:3], start[3:], goal_state[:3], n, tof)
        except ValueError:
            continue
        dv2 = goal_state[3:] - arrival
        total = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))
        after = start.copy()
        after[3:] += dv1
        coast, _ = _coast_min_distance(after, n, tof, 200)
        arrived = np.concatenate([goal_state[:3], arrival])
        missed, _ = _coast_min_distance(arrived, n, safety_orbits * period)
        candidate = (total, dv1, dv2, tof)
        if best_any is None or total < best_any[0]:
            best_any = candidate
        if min(coast, missed) >= keep_out_m and (best is None or total < best[0]):
            best = candidate
    chosen = best or best_any
    if chosen is None:
        raise ValueError("no two-impulse transfer on the time-of-flight grid")
    _, dv1, dv2, tof = chosen
    return dv1, dv2, tof


def _with_correction(
    burns: list[Burn], n: float, tof: float, t: float, aim: NDArray, label: str
) -> tuple[float, NDArray]:
    """Coast the transfer just planned, with a mid-course correction halfway."""
    half = tof / 2.0
    state = cw.propagate(burns[-1].state_after, n, half)
    burns.append(Burn(label, "transfer", t + half, np.zeros(3), state, half, aim))
    return t + tof, cw.propagate(state, n, half)


def design_inspection(
    n: float,
    hold_m: float = HOLD_M,
    waypoints_m: tuple[float, ...] = WAYPOINTS_M,
    ellipse_radial_m: float = ELLIPSE_RADIAL_M,
    ellipse_cross_track_m: float = ELLIPSE_CROSS_TRACK_M,
    keep_out_m: float = KEEP_OUT_M,
    inspection_orbits: int = INSPECTION_ORBITS,
    departure_m: float = DEPARTURE_M,
    safety_orbits: float = SAFETY_ORBITS,
) -> ProximityDesign:
    """Design the approach, inspection and departure for a target of mean motion n.

    Raises
    ------
    ValueError
        If the safety ellipse's radial or cross-track size is inside the
        keep-out radius, which would make it unsafe by design.
    """
    if min(ellipse_radial_m, ellipse_cross_track_m) <= keep_out_m:
        raise ValueError(
            "the safety ellipse must be wider than the keep-out sphere in both"
            " radial and cross-track"
        )
    period = 2.0 * np.pi / n
    burns: list[Burn] = []
    state = np.array([0.0, -hold_m, 0.0, 0.0, 0.0, 0.0])
    t = 0.0

    for distance in waypoints_m:
        hop = cw.radial_hop_speed(n, -distance - state[1])
        goal = np.array([0.0, -distance, 0.0])
        burns.append(
            Burn(
                f"Hop to {distance:,.0f} m",
                "hop",
                t,
                np.array([hop, 0.0, 0.0]) - state[3:],
                state,
                period / 2,
                goal,
            )
        )
        state = cw.propagate(burns[-1].state_after, n, period / 2)
        t += period / 2
        burns.append(Burn(f"Stop at {distance:,.0f} m", "brake", t, -state[3:], state))
        state = burns[-1].state_after

    entry = cw.safety_ellipse_state(ellipse_radial_m, ellipse_cross_track_m, n, np.pi)
    dv1, dv2, tof = _choose_transfer(state, entry, n, keep_out_m, safety_orbits)
    burns.append(
        Burn("Transfer to the ellipse", "transfer", t, dv1, state, tof, entry[:3])
    )
    t, state = _with_correction(burns, n, tof, t, entry[:3], "Correct course")
    burns.append(Burn("Enter the safety ellipse", "insert", t, dv2, state, 0, entry))
    state = burns[-1].state_after

    for k in range(1, inspection_orbits + 1):
        state = cw.propagate(state, n, period)
        t += period
        label = "Leave the ellipse" if k == inspection_orbits else f"Keep, orbit {k}"
        if k < inspection_orbits:
            burns.append(Burn(label, "keep", t, np.zeros(3), state, 0, entry))

    hold = np.array([0.0, -departure_m, 0.0, 0.0, 0.0, 0.0])
    dv1, dv2, tof = _choose_transfer(state, hold, n, keep_out_m, safety_orbits)
    burns.append(Burn("Leave the ellipse", "transfer", t, dv1, state, tof, hold[:3]))
    t, state = _with_correction(burns, n, tof, t, hold[:3], "Correct course out")
    burns.append(Burn(f"Stop at {departure_m:,.0f} m", "brake", t, dv2, state))
    return ProximityDesign(
        n=n,
        burns=tuple(burns),
        end_time_s=t + period,
        keep_out_m=keep_out_m,
        ellipse_radial_m=ellipse_radial_m,
        ellipse_cross_track_m=ellipse_cross_track_m,
        inspection_orbits=inspection_orbits,
        safety_orbits=safety_orbits,
    )


def missed_burn_checks(design: ProximityDesign) -> list[SafetyCheck]:
    """For every burn with a nominal speed change, coast from the state before
    it as if it failed, and report the closest approach to the target."""
    checks = []
    for k, burn in enumerate(design.burns):
        if burn.dv_m_s == 0.0:
            continue
        closest, when = _coast_min_distance(
            burn.state_before, design.n, design.safety_orbits * design.period_s
        )
        checks.append(
            SafetyCheck(k, burn.label, closest, when, closest >= design.keep_out_m)
        )
    return checks
