"""Rendezvous planning from a rideshare drop-off to a catalogued target.

The design problem: a rideshare leaves the spacecraft in an orbit whose plane
differs from the target's by some angle of node. Turning a plane with
propellant costs over 100 m/s per degree in low orbit, so a node difference of
ten or twenty degrees is out of reach. J2 turns planes for free, at a rate that
depends on altitude and inclination, so the plan is to move to a drift orbit
whose node precesses at a different rate from the target's, wait there until
the planes line up, and then transfer into the target's orbit. The choice of
drift orbit trades time against speed change, and that trade is the output.

Phasing along the orbit is free in this plan: the drift orbit laps the target
once per synodic period, so timing the final transfer within one synodic period
sets any arrival phase. The last step is a two-impulse Lambert transfer from a
far point behind the target to a hold point close behind it, where proximity
operations take over.

Mean, secular, first-order J2 dynamics; circular orbits; impulsive burns; no
drag. Units: km, km/s, radians, seconds, unless a name says otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from orbwatch.budget.budget import DeltaVItem
from orbwatch.catalog.elements import MU_EARTH_KM3_S2
from orbwatch.catalog.frames import J2000_JULIAN_DATE
from orbwatch.catalog.timescales import (
    SECONDS_PER_DAY,
    julian_date_split,
    require_utc,
)
from orbwatch.catalog.tle import TLE
from orbwatch.transfer.j2 import (
    EARTH_RADIUS_KM,
    nodal_rate_rad_s,
    sun_synchronous_inclination_rad,
)
from orbwatch.transfer.lambert import lambert
from orbwatch.transfer.manoeuvres import (
    Transfer,
    circular_speed_km_s,
    hohmann,
    orbit_plane_angle_rad,
    period_s,
    synodic_period_s,
)

MIN_DRIFT_ALTITUDE_KM: float = 400.0
"""Lowest drift orbit considered. Below it drag shortens a months-long wait
and the J2 model here, which ignores drag, stops being adequate."""

DRIFT_ALTITUDES_KM: tuple[float, ...] = tuple(np.arange(400.0, 1000.1, 10.0))
DRIFT_INCLINATION_OFFSETS_DEG: tuple[float, ...] = tuple(
    np.round(np.arange(-2.0, 2.001, 0.05), 3)
)

FAR_POINT_KM: float = 30.0
HOLD_POINT_KM: float = 5.0
"""Where the transfer arrives and where proximity operations begin, as
distances behind the target along its orbit."""


def _wrap_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class CircularOrbit:
    """A near-circular orbit: size, plane and the epoch the node refers to."""

    a_km: float
    inclination_rad: float
    raan_rad: float
    epoch_utc: datetime

    @property
    def altitude_km(self) -> float:
        return self.a_km - EARTH_RADIUS_KM

    @property
    def nodal_rate_rad_s(self) -> float:
        return nodal_rate_rad_s(self.a_km, self.inclination_rad)

    def raan_at(self, when_utc: datetime) -> float:
        """Node at another time, advanced at the secular J2 rate."""
        dt = (require_utc(when_utc) - self.epoch_utc).total_seconds()
        return float((self.raan_rad + self.nodal_rate_rad_s * dt) % (2.0 * np.pi))

    def at(self, when_utc: datetime) -> CircularOrbit:
        return CircularOrbit(
            self.a_km,
            self.inclination_rad,
            self.raan_at(when_utc),
            require_utc(when_utc),
        )


def orbit_from_tle(tle: TLE, mu_km3_s2: float = MU_EARTH_KM3_S2) -> CircularOrbit:
    """Mean orbit from an element set, treated as circular."""
    n_rad_s = tle.mean_motion_rev_per_day * 2.0 * np.pi / SECONDS_PER_DAY
    a_km = float((mu_km3_s2 / n_rad_s**2) ** (1.0 / 3.0))
    return CircularOrbit(a_km, tle.inclination_rad, tle.raan_rad, tle.epoch_utc)


def mean_sun_right_ascension_rad(when_utc: datetime) -> float:
    """Right ascension of the mean Sun, which moves uniformly along the equator.

    Mean longitude of the Sun, 280.460 + 0.9856474 deg per day since J2000
    (Astronomical Almanac). Local times of sun-synchronous orbits are defined
    against it: the true Sun runs up to 16 minutes ahead or behind through the
    year, the equation of time, and a sun-synchronous node follows the mean one.
    """
    whole, fraction = julian_date_split(require_utc(when_utc))
    days = (whole - J2000_JULIAN_DATE) + fraction
    return float(np.deg2rad((280.460 + 0.9856474 * days) % 360.0))


def local_time_of_ascending_node_h(orbit: CircularOrbit, when_utc: datetime) -> float:
    """Local mean solar time at which the orbit crosses the equator going north."""
    hour_angle = orbit.raan_at(when_utc) - mean_sun_right_ascension_rad(when_utc)
    return float((12.0 + np.rad2deg(hour_angle) / 15.0) % 24.0)


def sun_synchronous_orbit(
    altitude_km: float, ltan_h: float, when_utc: datetime
) -> CircularOrbit:
    """Circular sun-synchronous orbit with a given local time of ascending node."""
    a_km = EARTH_RADIUS_KM + altitude_km
    raan = mean_sun_right_ascension_rad(when_utc) + np.deg2rad((ltan_h - 12.0) * 15.0)
    return CircularOrbit(
        a_km,
        sun_synchronous_inclination_rad(a_km),
        float(raan % (2.0 * np.pi)),
        require_utc(when_utc),
    )


SUN_SYNCHRONOUS_FAMILY_DEG: float = 5.0
"""A target whose inclination is within this of sun-synchronous at its altitude
is reached from a sun-synchronous rideshare; any other from a rideshare into its
own inclination."""


def rideshare_orbit(
    target: CircularOrbit,
    altitude_km: float,
    node_offset_rad: float,
    when_utc: datetime,
) -> tuple[CircularOrbit, str]:
    """The drop-off a rideshare would realistically give for this target.

    Near-sun-synchronous targets, such as ENVISAT, are reached from a
    sun-synchronous rideshare, the most common kind: its plane differs from the
    target's by some node and a fraction of a degree of inclination. Any other
    target, such as the ISS at 51.6 deg, needs a rideshare into its own
    inclination, because J2 turns planes about the pole and cannot change
    inclination: from a sun-synchronous drop-off the ISS would cost some 6 km/s.

    Parameters
    ----------
    target : CircularOrbit
    altitude_km : float
        Drop-off altitude.
    node_offset_rad : float
        Drop-off node minus the target's, at ``when_utc``. For a sun-synchronous
        drop-off each hour of local time of ascending node is 15 deg.
    when_utc : datetime

    Returns
    -------
    orbit : CircularOrbit
    kind : {"sun-synchronous", "matched inclination"}
    """
    when = require_utc(when_utc)
    a_km = EARTH_RADIUS_KM + altitude_km
    raan = float((target.raan_at(when) + node_offset_rad) % (2.0 * np.pi))
    try:
        family = sun_synchronous_inclination_rad(target.a_km)
    except ValueError:
        family = None
    if family is not None and abs(np.rad2deg(target.inclination_rad - family)) <= (
        SUN_SYNCHRONOUS_FAMILY_DEG
    ):
        inclination, kind = sun_synchronous_inclination_rad(a_km), "sun-synchronous"
    else:
        inclination, kind = target.inclination_rad, "matched inclination"
    return CircularOrbit(a_km, inclination, raan, when), kind


@dataclass(frozen=True)
class DriftOption:
    """One way of closing the node gap: go to a drift orbit, wait, come back up.

    Attributes
    ----------
    altitude_km, inclination_rad : float
        The drift orbit.
    wait_days : float
        Time in the drift orbit until the planes line up.
    phasing_days : float
        One synodic period between the drift and target orbits: the longest
        extra wait needed to arrive at the right place along the orbit.
    enter, leave : Transfer
        From the drop-off into the drift orbit, and from it into the target's.
    """

    altitude_km: float
    inclination_rad: float
    wait_days: float
    phasing_days: float
    enter: Transfer
    leave: Transfer

    @property
    def dv_km_s(self) -> float:
        return self.enter.dv_km_s + self.leave.dv_km_s

    @property
    def total_days(self) -> float:
        return self.wait_days + self.phasing_days


def node_wait_s(delta_raan_rad: float, relative_rate_rad_s: float) -> float:
    """Time for a node gap to close at a relative rate, going round if needed.

    ``delta_raan_rad`` is target minus chaser; it closes when it reaches a
    multiple of a full turn.
    """
    if relative_rate_rad_s == 0.0:
        return float("inf")
    turn = 2.0 * np.pi / abs(relative_rate_rad_s)
    return float((-delta_raan_rad / relative_rate_rad_s) % turn)


def drift_options(
    dropoff: CircularOrbit,
    target: CircularOrbit,
    altitudes_km: Sequence[float] = DRIFT_ALTITUDES_KM,
    inclination_offsets_deg: Sequence[float] = DRIFT_INCLINATION_OFFSETS_DEG,
    min_altitude_km: float = MIN_DRIFT_ALTITUDE_KM,
) -> list[DriftOption]:
    """Every drift orbit on the grid, with its wait and its speed change.

    The drift inclination is the drop-off inclination plus an offset: tilting
    the drift orbit changes its nodal rate, often much faster than changing its
    altitude, at the price of undoing the tilt on the way out.
    """
    epoch = dropoff.epoch_utc
    gap = _wrap_pi(target.raan_at(epoch) - dropoff.raan_rad)
    options = []
    for altitude in altitudes_km:
        if altitude < min_altitude_km:
            continue
        a_drift = EARTH_RADIUS_KM + float(altitude)
        phasing = synodic_period_s(a_drift, target.a_km) / SECONDS_PER_DAY
        for offset in inclination_offsets_deg:
            inclination = dropoff.inclination_rad + np.deg2rad(offset)
            relative = target.nodal_rate_rad_s - nodal_rate_rad_s(a_drift, inclination)
            wait = node_wait_s(gap, relative) / SECONDS_PER_DAY
            options.append(
                DriftOption(
                    altitude_km=float(altitude),
                    inclination_rad=float(inclination),
                    wait_days=wait,
                    phasing_days=phasing,
                    enter=hohmann(
                        dropoff.a_km, a_drift, inclination - dropoff.inclination_rad
                    ),
                    leave=hohmann(
                        a_drift, target.a_km, target.inclination_rad - inclination
                    ),
                )
            )
    return options


def pareto_front(options: Sequence[DriftOption]) -> list[DriftOption]:
    """The options no other option beats on both time and speed change,
    fastest first."""
    front: list[DriftOption] = []
    for option in sorted(options, key=lambda o: (o.total_days, o.dv_km_s)):
        if not front or option.dv_km_s < front[-1].dv_km_s:
            front.append(option)
    return front


def direct_transfer(dropoff: CircularOrbit, target: CircularOrbit) -> Transfer:
    """Transfer straight into the target orbit, turning the whole plane by thrust."""
    angle = orbit_plane_angle_rad(
        dropoff.inclination_rad,
        dropoff.raan_rad,
        target.inclination_rad,
        target.raan_at(dropoff.epoch_utc),
    )
    return hohmann(dropoff.a_km, target.a_km, angle)


@dataclass(frozen=True)
class Approach:
    """Two-impulse Lambert transfer from the far point to the hold point."""

    far_km: float
    hold_km: float
    time_of_flight_s: float
    dv1_km_s: float
    dv2_km_s: float

    @property
    def dv_km_s(self) -> float:
        return self.dv1_km_s + self.dv2_km_s


def _circular_state(
    orbit: CircularOrbit, u_rad: float, mu_km3_s2: float = MU_EARTH_KM3_S2
) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity at argument of latitude ``u_rad`` on a circular orbit."""
    cos_o, sin_o = np.cos(orbit.raan_rad), np.sin(orbit.raan_rad)
    cos_i, sin_i = np.cos(orbit.inclination_rad), np.sin(orbit.inclination_rad)
    node = np.array([cos_o, sin_o, 0.0])
    normal = np.array([sin_o * sin_i, -cos_o * sin_i, cos_i])
    in_plane = np.cross(normal, node)
    radial = np.cos(u_rad) * node + np.sin(u_rad) * in_plane
    along = -np.sin(u_rad) * node + np.cos(u_rad) * in_plane
    return orbit.a_km * radial, circular_speed_km_s(orbit.a_km, mu_km3_s2) * along


def terminal_approach(
    target: CircularOrbit,
    far_km: float = FAR_POINT_KM,
    hold_km: float = HOLD_POINT_KM,
    orbit_fractions: Sequence[float] = tuple(np.linspace(0.2, 0.95, 76)),
) -> Approach:
    """Cheapest single-revolution Lambert transfer from far point to hold point.

    Both points trail the target along its orbit. The transfer is solved in the
    inertial frame for each time of flight on the grid, as a fraction of the
    target's period, and the cheapest is kept; flights that make the transfer
    angle close to 180 deg are skipped, since the plane is then undefined.
    """
    period = period_s(target.a_km)
    n = 2.0 * np.pi / period
    r1, v_start = _circular_state(target, -far_km / target.a_km)
    normal = np.cross(r1, v_start)
    best: Approach | None = None
    for fraction in orbit_fractions:
        tof = float(fraction) * period
        r2, v_hold = _circular_state(target, n * tof - hold_km / target.a_km)
        try:
            v1, v2 = lambert(r1, r2, tof, normal=normal)
        except ValueError:
            continue
        candidate = Approach(
            far_km=far_km,
            hold_km=hold_km,
            time_of_flight_s=tof,
            dv1_km_s=float(np.linalg.norm(v1 - v_start)),
            dv2_km_s=float(np.linalg.norm(v_hold - v2)),
        )
        if best is None or candidate.dv_km_s < best.dv_km_s:
            best = candidate
    if best is None:
        raise ValueError("no Lambert solution on the time-of-flight grid")
    return best


@dataclass(frozen=True)
class RendezvousPlan:
    """The whole transfer from drop-off to hold point, and the trade behind it.

    ``chosen`` is the cheapest drift option that fits within ``max_days``.
    ``front`` is the full time against speed-change trade.
    """

    dropoff: CircularOrbit
    target: CircularOrbit
    delta_raan_rad: float
    delta_inclination_rad: float
    max_days: float
    chosen: DriftOption
    front: tuple[DriftOption, ...]
    direct: Transfer
    approach: Approach

    @property
    def arrival_utc(self) -> datetime:
        leg_days = (
            self.chosen.enter.time_of_flight_s
            + self.chosen.leave.time_of_flight_s
            + self.approach.time_of_flight_s
        ) / SECONDS_PER_DAY
        return self.dropoff.epoch_utc + timedelta(
            days=self.chosen.total_days + leg_days
        )


def plan_rendezvous(
    dropoff: CircularOrbit,
    target: CircularOrbit,
    max_days: float = 180.0,
    **grid: Sequence[float],
) -> RendezvousPlan:
    """Plan the transfer, choosing the cheapest drift that fits the time budget.

    Raises
    ------
    ValueError
        If no drift option on the grid fits within ``max_days``.
    """
    options = drift_options(dropoff, target, **grid)
    front = pareto_front(options)
    feasible = [o for o in front if o.total_days <= max_days]
    if not feasible:
        raise ValueError(
            f"no drift orbit closes the node gap within {max_days:.0f} days;"
            f" the fastest on the grid takes {front[0].total_days:.0f}"
        )
    chosen = min(feasible, key=lambda o: o.dv_km_s)
    target_now = target.at(dropoff.epoch_utc)
    return RendezvousPlan(
        dropoff=dropoff,
        target=target_now,
        delta_raan_rad=_wrap_pi(target_now.raan_rad - dropoff.raan_rad),
        delta_inclination_rad=target.inclination_rad - dropoff.inclination_rad,
        max_days=max_days,
        chosen=chosen,
        front=tuple(front),
        direct=direct_transfer(dropoff, target),
        approach=terminal_approach(target_now),
    )


@dataclass(frozen=True)
class MissionAssumptions:
    """The numbers the budget needs that the trajectory does not decide.

    Defaults are assumptions for a small inspector, stated here so they are
    visible and easy to change, not taken from any specific vehicle.
    """

    injection_altitude_error_km: float = 10.0
    injection_inclination_error_deg: float = 0.1
    proximity_allocation_m_s: float = 10.0
    operations_days: float = 90.0
    maintenance_m_s_per_year: float = 2.0
    disposal_perigee_km: float = 300.0


def disposal_dv_km_s(a_km: float, perigee_altitude_km: float) -> float:
    """One burn lowering perigee from a circular orbit, for atmospheric re-entry."""
    r_perigee = EARTH_RADIUS_KM + perigee_altitude_km
    a_ellipse = 0.5 * (a_km + r_perigee)
    v_apogee = np.sqrt(MU_EARTH_KM3_S2 * (2.0 / a_km - 1.0 / a_ellipse))
    return float(circular_speed_km_s(a_km) - v_apogee)


def budget_items(
    plan: RendezvousPlan,
    assumptions: MissionAssumptions | None = None,
    proximity: tuple[float, str] | None = None,
) -> list[DeltaVItem]:
    """The delta-v lines of the mission, from drop-off to disposal.

    ``proximity`` is (delta-v, basis) from a designed and Monte Carlo'd
    proximity phase; without it the proximity line is the assumption's
    allocation, at the 100% margin of anything not yet designed.
    """
    assumptions = assumptions or MissionAssumptions()
    c = plan.chosen
    dispersion = hohmann(
        plan.dropoff.a_km,
        plan.dropoff.a_km + assumptions.injection_altitude_error_km,
        np.deg2rad(assumptions.injection_inclination_error_deg),
    )
    years = (
        (plan.arrival_utc - plan.dropoff.epoch_utc).total_seconds() / SECONDS_PER_DAY
        + assumptions.operations_days
    ) / 365.25
    return [
        DeltaVItem(
            "Injection dispersion correction",
            dispersion.dv_km_s * 1000.0,
            "dispersion",
            f"assumed {assumptions.injection_altitude_error_km:g} km,"
            f" {assumptions.injection_inclination_error_deg:g} deg",
        ),
        DeltaVItem(
            "Enter drift orbit",
            c.enter.dv_km_s * 1000.0,
            "calculated",
            f"Hohmann to {c.altitude_km:.0f} km with plane change",
        ),
        DeltaVItem(
            "Leave drift orbit into target orbit",
            c.leave.dv_km_s * 1000.0,
            "calculated",
            "Hohmann with plane change",
        ),
        DeltaVItem(
            "Far-range approach",
            plan.approach.dv_km_s * 1000.0,
            "calculated",
            f"Lambert, {plan.approach.far_km:g} to {plan.approach.hold_km:g} km behind",
        ),
        (
            DeltaVItem("Proximity operations", proximity[0], "calculated", proximity[1])
            if proximity
            else DeltaVItem(
                "Proximity operations",
                assumptions.proximity_allocation_m_s,
                "allocation",
                "allocation until designed",
            )
        ),
        DeltaVItem(
            "Orbit maintenance and collision avoidance",
            assumptions.maintenance_m_s_per_year * years,
            "maintenance",
            f"{assumptions.maintenance_m_s_per_year:g} m/s per year, {years:.2f} years",
        ),
        DeltaVItem(
            "Disposal",
            disposal_dv_km_s(plan.target.a_km, assumptions.disposal_perigee_km)
            * 1000.0,
            "calculated",
            f"perigee to {assumptions.disposal_perigee_km:g} km",
        ),
    ]
