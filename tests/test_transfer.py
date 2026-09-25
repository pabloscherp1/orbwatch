"""Tests for transfers: propagation, Lambert, manoeuvres, J2 drift, planning.

Reference values are textbook cases: Curtis example 5.2 for Lambert, the
classic 300 km to GEO Hohmann transfer, and well-known sun-synchronous and ISS
nodal figures. The planner is tested on its own consistency: the plan it picks
must actually line the planes up, and its trade must be a real trade.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from orbwatch.budget.budget import build_budget
from orbwatch.transfer.j2 import (
    EARTH_RADIUS_KM,
    SUN_SYNCHRONOUS_RATE_RAD_S,
    nodal_rate_rad_s,
    sun_synchronous_inclination_rad,
)
from orbwatch.transfer.kepler import propagate_kepler
from orbwatch.transfer.lambert import lambert
from orbwatch.transfer.manoeuvres import (
    hohmann,
    orbit_plane_angle_rad,
    plane_change_dv_km_s,
    synodic_period_s,
)
from orbwatch.transfer.mission import (
    CircularOrbit,
    budget_items,
    local_time_of_ascending_node_h,
    node_wait_s,
    plan_rendezvous,
    rideshare_orbit,
    sun_synchronous_orbit,
)

MU_CURTIS = 398600.0
EPOCH = datetime(2026, 9, 24, tzinfo=UTC)

# --------------------------------------------------------------------------
# Two-body propagation
# --------------------------------------------------------------------------


def test_a_circular_orbit_closes_after_one_period() -> None:
    r0 = np.array([7000.0, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU_CURTIS / 7000.0), 0.0])
    period = 2.0 * np.pi * np.sqrt(7000.0**3 / MU_CURTIS)
    r, v = propagate_kepler(r0, v0, period, MU_CURTIS)
    assert np.linalg.norm(r - r0) < 1e-6 and np.linalg.norm(v - v0) < 1e-9


@pytest.mark.parametrize("speed", [8.5, 12.0], ids=["ellipse", "hyperbola"])
def test_energy_and_angular_momentum_are_conserved(speed: float) -> None:
    r0 = np.array([7000.0, 1000.0, -500.0])
    v0 = np.array([0.5, speed, 1.0])
    r, v = propagate_kepler(r0, v0, 4000.0, MU_CURTIS)

    def energy(r_, v_):
        return v_ @ v_ / 2 - MU_CURTIS / np.linalg.norm(r_)

    assert energy(r, v) == pytest.approx(energy(r0, v0), abs=1e-9)
    assert np.allclose(np.cross(r, v), np.cross(r0, v0), rtol=1e-10)


def test_propagating_back_returns_to_the_start() -> None:
    r0, v0 = np.array([6800.0, 200.0, 50.0]), np.array([0.1, 7.6, 0.8])
    r1, v1 = propagate_kepler(r0, v0, 2500.0)
    r2, v2 = propagate_kepler(r1, v1, -2500.0)
    assert np.allclose(r2, r0, atol=1e-7) and np.allclose(v2, v0, atol=1e-10)


# --------------------------------------------------------------------------
# Lambert
# --------------------------------------------------------------------------


def test_lambert_matches_curtis_example_5_2() -> None:
    v1, v2 = lambert(
        [5000.0, 10000.0, 2100.0], [-14600.0, 2500.0, 7000.0], 3600.0, MU_CURTIS
    )
    assert np.allclose(v1, [-5.9925, 1.9254, 3.2456], atol=5e-4)
    assert np.allclose(v2, [-3.3125, -4.1966, -0.38529], atol=5e-4)


def test_lambert_solution_flies_to_the_target_when_propagated() -> None:
    r1 = np.array([7000.0, 0.0, 0.0])
    r2 = 7300.0 * np.array(
        [np.cos(2.2), np.sin(2.2) * np.cos(1.7), np.sin(2.2) * np.sin(1.7)]
    )
    normal = np.cross(r1, r2) * -1.0
    for tof in (1500.0, 3000.0, 4500.0):
        v1, v2 = lambert(r1, r2, tof, normal=normal)
        r, v = propagate_kepler(r1, v1, tof)
        assert np.linalg.norm(r - r2) < 1e-6 and np.linalg.norm(v - v2) < 1e-9
        assert np.dot(np.cross(r1, v1), normal) > 0, "respects the requested direction"


def test_lambert_near_half_an_orbit_approaches_the_hohmann_transfer() -> None:
    r1, r2 = 7000.0, 7500.0
    t = hohmann(r1, r2)
    angle = np.pi - 1e-3
    v1, _ = lambert(
        [r1, 0, 0], [r2 * np.cos(angle), r2 * np.sin(angle), 0], t.time_of_flight_s
    )
    assert np.linalg.norm(v1) - np.sqrt(398600.4418 / r1) == pytest.approx(
        t.dv1_km_s, abs=5e-4
    )


def test_lambert_refuses_undefined_or_impossible_problems() -> None:
    with pytest.raises(ValueError, match="plane is undefined"):
        lambert([7000.0, 0, 0], [-7000.0, 0, 0], 3000.0)
    with pytest.raises(ValueError, match="positive"):
        lambert([7000.0, 0, 0], [0, 7000.0, 0], 0.0)


# --------------------------------------------------------------------------
# Manoeuvres
# --------------------------------------------------------------------------


def test_hohmann_from_300_km_to_geo_matches_the_textbook() -> None:
    t = hohmann(6678.0, 42164.0)
    assert t.dv1_km_s == pytest.approx(2.426, abs=1e-3)
    assert t.dv2_km_s == pytest.approx(1.467, abs=1e-3)
    assert t.time_of_flight_s / 3600.0 == pytest.approx(5.28, abs=0.01)


def test_the_plane_change_goes_mostly_where_the_orbit_is_slowest() -> None:
    t = hohmann(6678.0, 42164.0, np.deg2rad(28.5))
    assert t.plane_split * 28.5 == pytest.approx(2.2, abs=0.2), "2.2 deg at perigee"
    assert t.dv_km_s == pytest.approx(4.23, abs=0.01)
    assert t.dv_km_s < hohmann(6678.0, 42164.0).dv_km_s + plane_change_dv_km_s(
        np.sqrt(398600.4418 / 42164.0), np.deg2rad(28.5)
    )


def test_plane_change_costs_and_plane_angles() -> None:
    assert plane_change_dv_km_s(7.5, np.deg2rad(1.0)) * 1000 == pytest.approx(
        130.9, abs=0.1
    )
    i = np.deg2rad(98.0)
    assert orbit_plane_angle_rad(i, 0.0, i, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert orbit_plane_angle_rad(i, 0.0, i + 0.01, 0.0) == pytest.approx(0.01)
    assert orbit_plane_angle_rad(np.pi / 2, 0.0, np.pi / 2, 0.3) == pytest.approx(0.3)


def test_synodic_period() -> None:
    days = synodic_period_s(EARTH_RADIUS_KM + 560.0, EARTH_RADIUS_KM + 761.0) / 86400
    assert 1.4 < days < 1.8
    assert synodic_period_s(7000.0, 7000.0) == float("inf")


# --------------------------------------------------------------------------
# J2
# --------------------------------------------------------------------------


def test_sun_synchronous_and_iss_nodal_rates() -> None:
    for altitude, inclination in ((525.0, 97.50), (800.0, 98.60)):
        a = EARTH_RADIUS_KM + altitude
        i = sun_synchronous_inclination_rad(a)
        assert np.rad2deg(i) == pytest.approx(inclination, abs=0.01)
        assert nodal_rate_rad_s(a, i) == pytest.approx(SUN_SYNCHRONOUS_RATE_RAD_S)
    iss = nodal_rate_rad_s(EARTH_RADIUS_KM + 420.0, np.deg2rad(51.64))
    assert np.rad2deg(iss) * 86400 == pytest.approx(-4.95, abs=0.05)
    with pytest.raises(ValueError):
        sun_synchronous_inclination_rad(EARTH_RADIUS_KM + 7000.0)


@pytest.mark.parametrize(
    ("gap_deg", "rate_deg_day", "days"),
    [(10.0, -1.0, 10.0), (10.0, 1.0, 350.0), (-10.0, 1.0, 10.0), (0.0, 1.0, 0.0)],
)
def test_node_gap_closes_at_the_relative_rate(
    gap_deg: float, rate_deg_day: float, days: float
) -> None:
    rate = np.deg2rad(rate_deg_day) / 86400.0
    assert node_wait_s(np.deg2rad(gap_deg), rate) / 86400 == pytest.approx(days)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def envisat_like() -> tuple[CircularOrbit, CircularOrbit]:
    target = CircularOrbit(
        EARTH_RADIUS_KM + 761.5, np.deg2rad(98.39), np.deg2rad(216.4), EPOCH
    )
    ltan = local_time_of_ascending_node_h(target, EPOCH) - 1.0
    return sun_synchronous_orbit(525.0, ltan, EPOCH), target


def test_a_sun_synchronous_drop_off_has_the_requested_local_time() -> None:
    orbit = sun_synchronous_orbit(525.0, 10.5, EPOCH)
    assert local_time_of_ascending_node_h(orbit, EPOCH) == pytest.approx(10.5, abs=1e-9)
    later = EPOCH + timedelta(days=100)
    assert local_time_of_ascending_node_h(orbit, later) == pytest.approx(10.5, abs=0.02)


def test_the_chosen_drift_lines_the_planes_up() -> None:
    dropoff, target = envisat_like()
    plan = plan_rendezvous(dropoff, target, max_days=180.0)
    c = plan.chosen
    assert c.total_days <= 180.0

    drift = CircularOrbit(
        EARTH_RADIUS_KM + c.altitude_km, c.inclination_rad, dropoff.raan_rad, EPOCH
    )
    aligned = EPOCH + timedelta(days=c.wait_days)
    gap = (target.raan_at(aligned) - drift.raan_at(aligned) + np.pi) % (
        2 * np.pi
    ) - np.pi
    assert abs(np.rad2deg(gap)) < 1e-6


def test_the_trade_is_a_real_trade_and_drifting_beats_turning() -> None:
    dropoff, target = envisat_like()
    plan = plan_rendezvous(dropoff, target, max_days=180.0)
    days = [o.total_days for o in plan.front]
    dv = [o.dv_km_s for o in plan.front]
    assert days == sorted(days) and dv == sorted(dv, reverse=True)
    assert plan.direct.dv_km_s > 5 * plan.chosen.dv_km_s, "a 15 deg node gap by thrust"
    assert plan_rendezvous(dropoff, target, max_days=60.0).chosen.dv_km_s > (
        plan.chosen.dv_km_s
    ), "hurrying costs propellant"

    floor = hohmann(
        dropoff.a_km, target.a_km, target.inclination_rad - dropoff.inclination_rad
    )
    assert min(dv) >= floor.dv_km_s - 1e-6, (
        "no drift beats the unavoidable raise and tilt"
    )


def test_the_rideshare_follows_the_target() -> None:
    """Near-sun-synchronous targets get a sun-synchronous drop-off, where 15 deg
    of node is one hour of local time; others get their own inclination."""
    _, envisat = envisat_like()
    dropoff, kind = rideshare_orbit(envisat, 525.0, np.deg2rad(-15.0), EPOCH)
    assert kind == "sun-synchronous"
    reference = sun_synchronous_orbit(
        525.0, local_time_of_ascending_node_h(envisat, EPOCH) - 1.0, EPOCH
    )
    assert dropoff.inclination_rad == pytest.approx(reference.inclination_rad)
    assert dropoff.raan_rad == pytest.approx(reference.raan_rad, abs=1e-9)

    iss = CircularOrbit(EARTH_RADIUS_KM + 419.0, np.deg2rad(51.63), 1.0, EPOCH)
    dropoff, kind = rideshare_orbit(iss, 525.0, np.deg2rad(-15.0), EPOCH)
    assert kind == "matched inclination"
    assert dropoff.inclination_rad == pytest.approx(iss.inclination_rad)
    plan = plan_rendezvous(dropoff, iss, max_days=180.0)
    assert plan.chosen.dv_km_s * 1000 < 150.0, "was 5.9 km/s from a sun-synchronous"


def test_an_impossible_deadline_is_refused_with_the_fastest_option() -> None:
    dropoff, target = envisat_like()
    with pytest.raises(ValueError, match="fastest on the grid"):
        plan_rendezvous(dropoff, target, max_days=10.0)


def test_the_approach_is_a_few_metres_per_second_and_the_budget_adds_up() -> None:
    dropoff, target = envisat_like()
    plan = plan_rendezvous(dropoff, target, max_days=180.0)
    assert 0.5 < plan.approach.dv_km_s * 1000 < 10.0

    items = budget_items(plan)
    labels = [i.label for i in items]
    assert "Disposal" in labels and "Injection dispersion correction" in labels
    budget = build_budget(items, 150.0, 220.0)
    transfer = next(i for i in items if i.label == "Enter drift orbit")
    assert transfer.dv_m_s == pytest.approx(plan.chosen.enter.dv_km_s * 1000)
    assert budget.dv_with_margins_m_s > budget.dv_nominal_m_s
