"""Tests for proximity operations.

The Clohessy-Wiltshire solution is checked against direct integration of its
own equations and against full nonlinear two-body motion; the frames by round
trip; the manoeuvres by their defining properties; the design by its safety
case; the Monte Carlo by reproducing the nominal plan with no errors.
"""

import numpy as np
import pytest

from orbwatch.rpo import cw, plan_proximity
from orbwatch.rpo.approach import design_inspection, missed_burn_checks
from orbwatch.rpo.frames import (
    curvilinear_to_inertial,
    inertial_to_curvilinear,
    inertial_to_ric,
    ric_to_inertial,
)
from orbwatch.rpo.monte_carlo import Dispersions, run_monte_carlo
from orbwatch.transfer.kepler import propagate_kepler

MU = 398600.4418
A_KM = 6378.137 + 761.5
N = float(np.sqrt(MU / A_KM**3))
PERIOD = 2 * np.pi / N
TARGET_R = np.array([A_KM, 0.0, 0.0])
TARGET_V = np.sqrt(MU / A_KM) * np.array([0.0, np.cos(1.717), np.sin(1.717)])
NO_ERRORS = Dispersions(0, 0, 0, 0, 0, 0, 0, 0)

# --------------------------------------------------------------------------
# Clohessy-Wiltshire
# --------------------------------------------------------------------------


def test_stm_is_the_identity_at_zero_and_composes() -> None:
    assert np.allclose(cw.stm(N, 0.0), np.eye(6))
    assert np.allclose(cw.stm(N, 1500.0) @ cw.stm(N, 2500.0), cw.stm(N, 4000.0))


def test_stm_solves_the_clohessy_wiltshire_equations() -> None:
    state = np.array([120.0, -800.0, 60.0, 0.05, -0.1, 0.02])
    dt, steps = 1.0, 3000
    x = state.copy()
    for _ in range(steps):

        def f(s):
            return np.array(
                [
                    s[3],
                    s[4],
                    s[5],
                    3 * N**2 * s[0] + 2 * N * s[4],
                    -2 * N * s[3],
                    -(N**2) * s[2],
                ]
            )

        k1 = f(x)
        k2 = f(x + dt / 2 * k1)
        k3 = f(x + dt / 2 * k2)
        k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.allclose(x, cw.propagate(state, N, dt * steps), atol=1e-6)


def test_propagate_handles_many_states_and_many_times() -> None:
    states = np.random.default_rng(0).normal(size=(7, 6))
    times = np.linspace(0, 3000, 5)
    out = cw.propagate(states, N, times)
    assert out.shape == (5, 7, 6)
    assert np.allclose(out[3, 2], cw.propagate(states[2], N, times[3]))


@pytest.mark.parametrize(
    ("state", "curvilinear_tolerance_m", "rectilinear_error_m"),
    [
        ([0, -5000.0, 0, 0, 0, 0], 1e-6, 60.0),
        ([0, -5000.0, 0, -N * 3500 / 4, 0, 0], 2.0, 50.0),
        ([100, -2000.0, 300, 0.1, -0.2, 0.05], 0.5, 10.0),
    ],
    ids=["5 km hold", "5 km radial hop", "generic 2 km"],
)
def test_curvilinear_clohessy_wiltshire_matches_two_body_motion(
    state: list[float], curvilinear_tolerance_m: float, rectilinear_error_m: float
) -> None:
    """Regression: in straight-line coordinates a 5 km V-bar hold drifts 66 m
    per orbit in the model; in curvilinear ones it stays put, as it should."""
    rel0 = np.array(state)
    worst = {"curvilinear": 0.0, "rectilinear": 0.0}
    for name, to_i, to_r in (
        ("curvilinear", curvilinear_to_inertial, inertial_to_curvilinear),
        ("rectilinear", ric_to_inertial, inertial_to_ric),
    ):
        rc, vc = to_i(TARGET_R, TARGET_V, rel0)
        for t in np.linspace(0, PERIOD, 13)[1:]:
            rt1, vt1 = propagate_kepler(TARGET_R, TARGET_V, t)
            rc1, vc1 = propagate_kepler(rc, vc, t)
            error = to_r(rt1, vt1, rc1, vc1)[:3] - cw.propagate(rel0, N, t)[:3]
            worst[name] = max(worst[name], float(np.linalg.norm(error)))
    assert worst["curvilinear"] < curvilinear_tolerance_m
    assert worst["rectilinear"] > rectilinear_error_m, "premise: straight-line is worse"


def test_frames_round_trip() -> None:
    rel = np.array([35.0, -700.0, 120.0, 0.02, -0.05, 0.01])
    for to_i, to_r in (
        (ric_to_inertial, inertial_to_ric),
        (curvilinear_to_inertial, inertial_to_curvilinear),
    ):
        assert np.allclose(
            to_r(TARGET_R, TARGET_V, *to_i(TARGET_R, TARGET_V, rel)), rel
        )


def test_a_radial_hop_arrives_in_half_an_orbit_and_returns_if_not_stopped() -> None:
    start = np.array([0.0, -5000.0, 0.0, cw.radial_hop_speed(N, 3500.0), 0.0, 0.0])
    half = cw.propagate(start, N, PERIOD / 2)
    assert half[:3] == pytest.approx([0.0, -1500.0, 0.0], abs=1e-6)
    assert cw.propagate(start, N, PERIOD)[:3] == pytest.approx(start[:3], abs=1e-6)


def test_the_safety_ellipse_is_drift_free_and_never_crosses_the_in_track_axis() -> None:
    for phase in np.linspace(0, 2 * np.pi, 9):
        state = cw.safety_ellipse_state(250.0, 300.0, N, phase)
        assert cw.in_track_drift_per_orbit(state, N) == pytest.approx(0.0, abs=1e-9)
    orbit = cw.propagate(
        cw.safety_ellipse_state(250.0, 300.0, N, 0.3), N, np.linspace(0, PERIOD, 400)
    )
    radial_cross = np.hypot(orbit[:, 0], orbit[:, 2])
    assert radial_cross.min() == pytest.approx(250.0, rel=1e-3), "min(A, B)"


def test_two_impulse_lands_on_the_aim_point_and_refuses_half_orbits() -> None:
    r0, v0, goal = np.array([0, -1500.0, 0]), np.zeros(3), np.array([0, -500.0, -250.0])
    dv1, arrival = cw.two_impulse(r0, v0, goal, N, 0.3 * PERIOD)
    end = cw.propagate(np.concatenate([r0, v0 + dv1]), N, 0.3 * PERIOD)
    assert np.allclose(end[:3], goal, atol=1e-6) and np.allclose(end[3:], arrival)
    with pytest.raises(ValueError, match="half orbits"):
        cw.two_impulse(r0, v0, goal, N, 0.5 * PERIOD)


# --------------------------------------------------------------------------
# Design and Monte Carlo
# --------------------------------------------------------------------------


def test_every_burn_can_fail_safely_in_the_nominal_design() -> None:
    design = design_inspection(N)
    checks = missed_burn_checks(design)
    assert checks and all(c.passes for c in checks)
    _, states = design.trajectory()
    assert np.linalg.norm(states[:, :3], axis=1).min() >= design.keep_out_m
    assert 2.0 < design.dv_m_s < 5.0


def test_an_ellipse_inside_the_keep_out_sphere_is_refused() -> None:
    with pytest.raises(ValueError, match="wider than the keep-out"):
        design_inspection(N, ellipse_radial_m=150.0)


def test_with_no_errors_the_monte_carlo_flies_the_nominal_plan() -> None:
    design = design_inspection(N)
    result = run_monte_carlo(design, NO_ERRORS, runs=5)
    assert result.dv_m_s == pytest.approx(np.full(5, design.dv_m_s))
    assert result.min_distance_m.min() == pytest.approx(250.0, abs=0.5)


def test_with_typical_errors_the_approach_stays_safe_and_cheap() -> None:
    plan = plan_proximity(A_KM, runs=1000)
    mc = plan.monte_carlo
    assert mc.violation_fraction < 0.01
    assert plan.design.dv_m_s < plan.budget_dv_m_s < 1.5 * plan.design.dv_m_s
    assert np.percentile(mc.min_distance_m, 1) > plan.design.keep_out_m


def test_isotropic_navigation_errors_would_have_broken_it() -> None:
    """Regression: 1% of range on every axis put radial errors of tens of metres
    into transfers that amplify them some 35 times in-track."""
    design = design_inspection(N)
    good = run_monte_carlo(design, Dispersions(), runs=1000)
    isotropic = run_monte_carlo(design, Dispersions(bearing_deg=0.573), runs=1000)
    assert isotropic.violation_fraction > 10 * max(good.violation_fraction, 0.001)
