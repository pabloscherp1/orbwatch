"""Tests for east-west pattern of life.

Synthetic satellites are flown through the station-keeping cycle with a known
policy: pulled east at a constant rate, burned back west whenever they reach
the east edge of the box, aimed so the turn-around lands on the west edge. The
estimator has to recover that policy and the predictors have to forecast it.
"""

from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from orbwatch.events import detect_manoeuvres
from orbwatch.events.detect import Manoeuvre
from orbwatch.events.pattern import (
    J22_AMPLITUDE_DEG_PER_DAY2,
    backtest_east_west,
    east_west_burns,
    estimate_east_west_policy,
    pattern_of_life,
    triaxiality_acceleration_deg_per_day2,
    weekday_counts,
)
from tests.test_events import T0, leo, make_history, sample_times

A_GEO_KM = 42164.17
SIDEREAL_DEG_PER_DAY = 360.9856


def station_kept(
    rng: np.random.Generator,
    days: float = 240.0,
    accel: float = 0.0015,
    west: float = -137.22,
    east: float = -137.18,
    west_jitter: float = 0.0,
    east_jitter: float = 0.0,
):
    """A GEO satellite in an east-west box, sampled like the real catalogue.

    Returns the history and the true burn times, in days.
    """
    step = 0.005
    grid = np.arange(0.0, days + 1.0, step)
    lon = np.empty_like(grid)
    rate = np.empty_like(grid)
    lam, lam_dot = east, -np.sqrt(2.0 * accel * (east - west))
    trigger = east
    burns = []
    for k in range(grid.size):
        lon[k], rate[k] = lam, lam_dot
        if lam >= trigger and lam_dot > 0.0:
            aim = west + west_jitter * rng.normal()
            lam_dot = -np.sqrt(2.0 * accel * max(lam - aim, 1e-4))
            trigger = east + east_jitter * rng.normal()
            burns.append(grid[k])
        lam += lam_dot * step + 0.5 * accel * step**2
        lam_dot += accel * step

    t = sample_times(days, rng)
    idx = np.searchsorted(grid, t)
    a = A_GEO_KM - (2.0 / 3.0) * A_GEO_KM * rate[idx] / SIDEREAL_DEG_PER_DAY
    a = a + rng.normal(0.0, 0.008, t.size)
    longitude = np.deg2rad(lon[idx] + rng.normal(0.0, 0.0003, t.size))
    history = make_history(
        t, a, np.zeros(t.size), np.zeros((t.size, 2)), "geosynchronous"
    )
    return replace(history, mean_longitude_rad=longitude), np.array(burns)


def test_triaxiality_is_zero_at_the_stable_points_and_pushes_towards_them() -> None:
    assert J22_AMPLITUDE_DEG_PER_DAY2 == pytest.approx(0.0017, abs=0.00005)
    for stable in (75.07, -104.93):
        assert abs(triaxiality_acceleration_deg_per_day2(stable)) < 1e-12
        assert triaxiality_acceleration_deg_per_day2(stable + 1.0) < 0.0
        assert triaxiality_acceleration_deg_per_day2(stable - 1.0) > 0.0
    assert triaxiality_acceleration_deg_per_day2(-137.2) == pytest.approx(
        0.00154, abs=2e-5
    )


def test_policy_recovers_the_cycle_it_was_flown_with() -> None:
    history, true_burns = station_kept(np.random.default_rng(40))
    burns = east_west_burns(detect_manoeuvres(history))
    assert abs(len(burns) - true_burns.size) <= 1

    policy = estimate_east_west_policy(history, burns)
    assert policy is not None
    period = 2.0 * np.sqrt(2.0 * 0.0015 * 0.04) / 0.0015
    assert policy.interval_days == pytest.approx(period, rel=0.05)
    assert policy.acceleration_deg_per_day2 == pytest.approx(0.0015, rel=0.1)
    assert policy.trigger_longitude_deg == pytest.approx(-137.18, abs=0.004)
    assert policy.box_deg[0] == pytest.approx(-137.22, abs=0.005)
    assert policy.box_deg[1] == pytest.approx(-137.18, abs=0.005)
    assert policy.delta_a_km > 0.0, "burns against an eastward pull raise the orbit"


def test_trigger_beats_interval_when_the_cycle_length_varies() -> None:
    """The operator aims at a different west edge each cycle, so intervals vary,
    but always burns at the same east edge, which the trigger model knows."""
    history, _ = station_kept(np.random.default_rng(41), days=300.0, west_jitter=0.012)
    backtest = backtest_east_west(history, detect_manoeuvres(history))
    assert backtest is not None and len(backtest.entries) >= 5

    trigger = backtest.score("trigger")
    interval = backtest.score("interval")
    assert trigger.median_abs_days < 0.5
    assert trigger.median_abs_days < interval.median_abs_days
    pattern = pattern_of_life(history, detect_manoeuvres(history))
    assert pattern.applies and pattern.preferred == "trigger"


def test_scatter_in_the_trigger_sets_the_timing_floor() -> None:
    history, _ = station_kept(np.random.default_rng(42), days=300.0, east_jitter=0.006)
    policy = estimate_east_west_policy(history, detect_manoeuvres(history))
    assert policy is not None
    expected = 0.006 / np.sqrt(2.0 * 0.0015 * 0.04)
    assert policy.timing_floor_days == pytest.approx(expected, rel=0.6)


def test_no_pattern_without_geo_or_without_enough_burns() -> None:
    rng = np.random.default_rng(43)
    t, a = leo(rng)
    assert (
        pattern_of_life(make_history(t, a), detect_manoeuvres(make_history(t, a)))
        is None
    )

    history, _ = station_kept(rng, days=40.0)
    verdict = pattern_of_life(history, detect_manoeuvres(history))
    assert verdict is not None and not verdict.applies
    assert "a pattern needs 6" in verdict.reason and verdict.forecast is None


def test_a_continuously_steered_satellite_gets_a_verdict_not_a_forecast() -> None:
    """Regression: INMARSAT 5-F3, steered by daily ion thruster firings, showed
    a pattern built on the lumps in its control that missed by three weeks.
    Here the arcs bend at a seventh of what gravity at that longitude gives."""
    history, _ = station_kept(
        np.random.default_rng(45), days=500.0, accel=0.0002, west=-137.22
    )
    pattern = pattern_of_life(history, detect_manoeuvres(history))
    assert pattern is not None and pattern.burns >= 6
    assert pattern.policy.acceleration_deg_per_day2 == pytest.approx(0.0002, rel=0.2)
    assert not pattern.applies
    assert "does not drift freely" in pattern.reason


def test_a_freely_drifting_satellite_with_skill_gets_a_forecast() -> None:
    history, _ = station_kept(np.random.default_rng(46), days=300.0)
    pattern = pattern_of_life(history, detect_manoeuvres(history))
    assert pattern is not None and pattern.applies and pattern.reason == ""
    assert pattern.forecast is not None


def burn_at(start_hours: float, width_hours: float) -> Manoeuvre:
    start = T0 + timedelta(hours=start_hours)
    return Manoeuvre(
        kind="in_plane",
        start_utc=start,
        end_utc=start + timedelta(hours=width_hours),
        change=(0.5,),
        delta_v_m_s=0.02,
        significance=100.0,
        gaps=1,
        coherence=1.0,
        profile="impulsive",
    )


def test_weekday_counts_use_well_timed_windows_only() -> None:
    assert T0.weekday() == 3, "T0 is a Thursday"
    burns = [
        burn_at(4 * 24 + 1, 10),  # Monday
        burn_at(11 * 24 + 2, 8),  # Monday
        burn_at(13 * 24, 12),  # Wednesday
        burn_at(20 * 24, 72),  # three days wide, cannot be placed
    ]
    assert weekday_counts(burns) == (2, 0, 1, 0, 0, 0, 0)
