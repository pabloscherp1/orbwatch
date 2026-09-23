"""Tests for the events layer: cleaning, manoeuvre detection and budgets.

Almost everything here runs on synthetic histories with known answers: burns of
known size at known times, bad element sets, multi-day bad runs, drag surges and
continuous control, on top of realistic noise. The detector must recover what
was injected and nothing else.

Three tests pin bugs found on real data during development, so they cannot come
back: a centred median rejecting genuine post-burn points, an isolated bad fit
counted as two burns, and summing net displacement across opposing burns.

The real-data validation against NASA's announced ISS reboosts runs as a
network test when Space-Track credentials are available.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from orbwatch.catalog.tle import parse_tle
from orbwatch.events import (
    OrbitHistory,
    analyse,
    analyse_steps,
    detect_drag_surges,
    detect_manoeuvres,
    merge_refits,
    natural_inclination_path_rad,
    natural_inclination_rate_rad_per_day,
    two_sided_transients,
)
from orbwatch.events.budget import in_plane_budget, out_of_plane_budget
from orbwatch.events.detect import FLOOR_OVERRIDE_SIGMAS, MIN_COHERENCE, manoeuvres_from
from orbwatch.events.history import DataQuality, robust_sigma
from tests.test_tle import build_line1, build_line2

T0 = datetime(2026, 1, 1, tzinfo=UTC)
ARCSEC = np.deg2rad(1.0 / 3600.0)
MU = 398600.8


def sample_times(
    days: float, rng: np.random.Generator, silences: tuple[float, ...] = ()
) -> np.ndarray:
    """Irregular sampling like the real catalogue: every 4 to 8 hours, with an
    optional 17-hour silence starting at each time in ``silences``."""
    t, out = 0.0, [0.0]
    while t < days:
        step = rng.uniform(4.0, 8.0) / 24.0
        if any(s <= t < s + step for s in silences):
            step = 17.0 / 24.0
        t += step
        out.append(t)
    return np.array(out)


def make_history(
    t, a_km, inclination_rad=None, ivec=None, regime="other"
) -> OrbitHistory:
    n = len(t)
    inc = (
        np.full(n, np.deg2rad(51.6))
        if inclination_rad is None
        else np.asarray(inclination_rad)
    )
    iv = np.zeros((n, 2)) if ivec is None else np.asarray(ivec)
    quality = DataQuality(n, n, 0, (), 60.0)
    return OrbitHistory(
        norad_id=99999,
        epochs_utc=tuple(T0 + timedelta(days=float(x)) for x in t),
        t_days=np.asarray(t, dtype=float),
        a_km=np.asarray(a_km, dtype=float),
        inclination_rad=inc,
        inclination_vector_rad=iv,
        regime=regime,
        quality=quality,
    )


def leo(
    rng,
    burns=((20.0, 1.0), (55.0, 0.4), (90.0, 2.5)),
    days=120.0,
    noise_km=0.005,
    silences=(),
):
    """ISS-like: 55 m/day drag decay, 5 m noise, reboosts of known size."""
    t = sample_times(days, rng, silences)
    a = 6800.0 - 0.055 * t + rng.normal(0.0, noise_km, t.size)
    for when, size in burns:
        a[t > when] += size
    return t, a


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


def test_robust_sigma_ignores_outliers() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 2000)
    assert robust_sigma(x) == pytest.approx(1.0, abs=0.08)
    x[:100] = 100.0
    assert robust_sigma(x) == pytest.approx(1.0, abs=0.15)


def test_refits_at_the_same_epoch_are_merged() -> None:
    line2 = build_line2()
    first = parse_tle(build_line1(epoch_day=250.0), line2)
    refit = parse_tle(build_line1(epoch_day=250.00000010), line2)
    later = parse_tle(build_line1(epoch_day=250.00200000), line2)

    assert len(merge_refits([first, refit])) == 1
    assert len(merge_refits([first, later])) == 2
    assert merge_refits([refit, first])[0].epoch_utc == refit.epoch_utc


def test_isolated_bad_fit_is_flagged_as_transient() -> None:
    rng = np.random.default_rng(2)
    t = sample_times(60.0, rng)
    x = 0.01 * t + rng.normal(0.0, 1.0, t.size)
    x[100] += 30.0
    assert np.flatnonzero(two_sided_transients(t, x)).tolist() == [100]


def test_multi_day_bad_run_is_flagged() -> None:
    rng = np.random.default_rng(3)
    t = sample_times(60.0, rng)
    x = rng.normal(0.0, 1.0, t.size)
    run = (t > 25.0) & (t < 28.0)
    x[run] += 30.0
    assert np.array_equal(two_sided_transients(t, x), run)


def test_genuine_step_after_a_silence_is_kept() -> None:
    """Regression: after the 13 March 2026 ISS reboost the catalogue went quiet
    for 17 hours, and a centred running median rejected the correct post-burn
    element sets. The two-sided test must keep them."""
    rng = np.random.default_rng(4)
    t = sample_times(60.0, rng, silences=(30.0,))
    x = rng.normal(0.0, 1.0, t.size)
    x[t > 30.2] += 40.0
    assert not two_sided_transients(t, x).any()


def test_clean_data_loses_almost_nothing_to_the_transient_test() -> None:
    """The threshold errs on the side of rejecting, by design. On clean data
    with a steady drift, about 1 good sample in 1,600 is rejected."""
    rejected = total = 0
    for seed in range(20):
        rng = np.random.default_rng(500 + seed)
        t = sample_times(90.0, rng)
        x = -55.0 * t + rng.normal(0.0, 1.0, t.size)
        rejected += int(two_sided_transients(t, x).sum())
        total += t.size
    assert rejected / total < 0.002


def test_transient_test_works_on_vectors_and_short_series() -> None:
    rng = np.random.default_rng(6)
    t = sample_times(40.0, rng)
    v = rng.normal(0.0, 1.0, (t.size, 2))
    v[50] += [20.0, -20.0]
    assert np.flatnonzero(two_sided_transients(t, v)).tolist() == [50]
    assert not two_sided_transients(np.arange(5.0), np.zeros(5)).any()


def test_a_bad_run_that_survives_cleaning_is_not_two_burns() -> None:
    """Regression: INMARSAT 5-F3 sat at zero inclination for several days around
    1 January 2026, and the run's entry and exit came out as two separate,
    individually coherent steps. Cleaning is switched off here to force the
    situation; the pair must be recognised as an excursion."""
    rng = np.random.default_rng(14)
    t, a = leo(rng, burns=((80.0, 1.0),))
    a[(t > 30.0) & (t < 36.0)] -= 0.8
    history = make_history(t, a)

    candidates = manoeuvres_from(history, analyse_steps(history, "in_plane"))
    assert sum(m.excursion for m in candidates) == 2
    found = [m for m in detect_manoeuvres(history) if m.kind == "in_plane"]
    assert len(found) == 1 and found[0].delta_a_km == pytest.approx(1.0, abs=0.03)


# --------------------------------------------------------------------------
# Low orbit, in-plane
# --------------------------------------------------------------------------


def test_reboosts_are_recovered_with_size_timing_and_delta_v() -> None:
    rng = np.random.default_rng(10)
    burns = ((20.0, 1.0), (55.0, 0.4), (90.0, 2.5))
    t, a = leo(rng, burns, silences=tuple(b[0] for b in burns))
    history = make_history(t, a)

    found = [m for m in detect_manoeuvres(history) if m.kind == "in_plane"]
    assert len(found) == 3
    for m, (when, size) in zip(found, burns, strict=True):
        assert m.delta_a_km == pytest.approx(size, abs=0.03)
        assert m.start_utc <= T0 + timedelta(days=when) <= m.end_utc
        assert m.profile == "impulsive" and m.gaps == 1 and m.coherence > 0.99
        v = np.sqrt(MU / a.mean()) * 1000.0
        assert m.delta_v_m_s == pytest.approx(0.5 * v * size / a.mean(), rel=0.03)


def test_low_orbit_budget_closes_when_every_reboost_is_seen() -> None:
    rng = np.random.default_rng(11)
    t, a = leo(rng)
    history = make_history(t, a)
    analysis = analyse_steps(history, "in_plane")
    budget = in_plane_budget(history, analysis, manoeuvres_from(history, analysis))
    assert budget.closure == pytest.approx(1.0, abs=0.04)


def test_an_isolated_bad_fit_is_never_counted_as_burns() -> None:
    """Regression: on the ISS a fit 45 m low produced two opposing 5-sigma
    steps. Without cleaning they must at least come out incoherent."""
    rng = np.random.default_rng(12)
    t, a = leo(rng, burns=())
    k = int(np.searchsorted(t, 40.0))
    a[k] -= 1.5
    history = make_history(t, a)

    candidates = manoeuvres_from(history, analyse_steps(history, "in_plane"))
    assert candidates, "the spike should produce raw detections"
    assert all(m.coherence < MIN_COHERENCE for m in candidates)
    assert detect_manoeuvres(history) == []


def test_fast_decay_at_the_start_of_a_window_is_not_a_burn() -> None:
    """Regression: the ISS year to 2026-09-22 opened with a 40-hour silence
    while the orbit decayed about three times faster than its yearly median, and
    the first gap, with no past to estimate the rate from, was flagged as a
    154 m lowering burn. A reboost right after the edge must still be found."""
    rng = np.random.default_rng(12)
    t = np.concatenate([[0.0], 40.0 / 24.0 + sample_times(120.0, rng)])
    phase = 2.0 * np.pi * t / 27.0
    rate = 0.09 + 0.06 * np.cos(phase)
    decay = np.concatenate(
        [[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))]
    )
    a = 6793.2 - decay + rng.normal(0.0, 0.005, t.size)
    reboost_at = 2.2
    a[t > reboost_at] += 3.5
    history = make_history(t, a)

    events = detect_manoeuvres(history)
    assert not [m for m in events if m.start_utc == history.epochs_utc[0]]
    reboosts = [m for m in events if m.delta_a_km and m.delta_a_km > 3.0]
    assert len(reboosts) == 1
    assert reboosts[0].start_utc <= T0 + timedelta(days=reboost_at)

    analysis = analyse_steps(history, "in_plane")
    assert analysis.edge[0], "the first gap has no past"
    assert analysis.magnitude[0] > analysis.threshold, "premise: plain 5 sigma flags it"
    assert analysis.magnitude[0] < analysis.gap_threshold[0]
    assert not analysis.edge[analysis.dt_days.size // 2]


def test_a_small_burn_on_a_clean_history_is_kept_despite_the_floor() -> None:
    """Regression: NOAA 20 scatters 0.1 m per gap, and its 57 m raise of
    28 January 2026, 0.03 m/s at 537 sigma, fell under the 0.05 m/s floor."""
    rng = np.random.default_rng(15)
    t = sample_times(120.0, rng)
    a = 7205.0 - 0.0016 * t + rng.normal(0.0, 0.0001, t.size)
    a[t > 60.0] += 0.057
    history = make_history(t, a, np.full(t.size, np.deg2rad(98.76)))

    burns = [m for m in detect_manoeuvres(history) if m.kind == "in_plane"]
    assert len(burns) == 1
    assert burns[0].delta_v_m_s < 0.05, "premise: under the floor"
    assert burns[0].significance > FLOOR_OVERRIDE_SIGMAS["in_plane"]
    assert burns[0].delta_a_km == pytest.approx(0.057, abs=0.002)


def test_a_modest_blip_on_a_noisy_history_still_falls_under_the_floor() -> None:
    rng = np.random.default_rng(16)
    t, a = leo(rng, burns=())
    k = t.size // 2
    a[k:] -= 0.075
    history = make_history(t, a)

    analysis = analyse_steps(history, "in_plane")
    events = [m for m in manoeuvres_from(history, analysis) if m.coherence >= 0.5]
    assert events and all(m.significance < 30.0 for m in events)
    assert all(m.delta_v_m_s < 0.05 for m in events)
    assert not [m for m in detect_manoeuvres(history) if m.kind == "in_plane"]


def test_rounded_inclination_never_gives_zero_scatter() -> None:
    """Regression: NOAA 20's inclination drifts one TLE digit a day or so, most
    gaps round to no change, and the robust scatter came out as exactly zero."""
    rng = np.random.default_rng(17)
    t = sample_times(120.0, rng)
    inclination_deg = np.round(98.76 + 0.00008 * t, 4)
    history = make_history(
        t,
        7205.0 - 0.0016 * t + rng.normal(0.0, 0.0001, t.size),
        np.deg2rad(inclination_deg),
    )

    analysis = analyse_steps(history, "out_of_plane")
    assert analysis.sigma >= np.deg2rad(1e-4) / np.sqrt(6.0) * 0.999
    assert not analysis.flagged.any(), "one rounding step is not a manoeuvre"


def test_a_drag_surge_is_natural_not_a_manoeuvre_and_counts_as_drag() -> None:
    """Regression: the G4 storm of 19 January 2026 sped up the decay of every
    low orbit at once, Hubble included, and was listed as a Hubble manoeuvre."""
    rng = np.random.default_rng(13)
    t, a = leo(rng, burns=((30.0, 1.5), (90.0, 0.8)))
    surge = (t > 60.0) & (t < 62.5)
    # Extra decay of 0.3 km/day for 2.5 days, like the ISS on 19 to 22 January
    # 2026, which lost an extra 191 m.
    a -= np.where(t > 60.0, np.minimum(t - 60.0, 2.5) * 0.3, 0.0)
    history = make_history(t, a)

    burns = [m for m in detect_manoeuvres(history) if m.kind == "in_plane"]
    assert surge.any()
    assert all(m.delta_a_km > 0 for m in burns) and len(burns) == 2

    surges = detect_drag_surges(history)
    assert len(surges) == 1
    assert surges[0].profile == "sustained" and surges[0].gaps > 1
    assert surges[0].delta_a_km == pytest.approx(-0.75, abs=0.08)

    analysis = analyse_steps(history, "in_plane")
    budget = in_plane_budget(history, analysis, burns, surges)
    assert budget.detected_m_s == pytest.approx(sum(m.delta_v_m_s for m in burns))
    assert budget.closure == pytest.approx(1.0, abs=0.05), "the surge is drag"
    without = in_plane_budget(history, analysis, burns)
    assert without.closure > 1.25, "premise: ignoring the surge overstates closure"


def test_a_sustained_raise_is_still_a_manoeuvre() -> None:
    rng = np.random.default_rng(14)
    t, a = leo(rng, burns=())
    a += np.where(t > 40.0, np.minimum(t - 40.0, 3.0) * 0.4, 0.0)
    history = make_history(t, a)

    raises = [m for m in detect_manoeuvres(history) if m.kind == "in_plane"]
    assert len(raises) == 1 and raises[0].profile == "sustained"
    assert not detect_drag_surges(history)


# --------------------------------------------------------------------------
# Geostationary, out-of-plane
# --------------------------------------------------------------------------


def natural_push_at_origin() -> np.ndarray:
    return natural_inclination_rate_rad_per_day(np.zeros(2))


def geo(rng, t, ivec):
    a = 42164.0 + rng.normal(0.0, 0.3, t.size)
    return make_history(
        t, a, np.linalg.norm(ivec, axis=1), ivec, regime="geosynchronous"
    )


def test_natural_drift_model_at_zero_inclination() -> None:
    rate = natural_push_at_origin()
    per_year_deg = np.rad2deg(np.linalg.norm(rate)) * 365.25
    assert per_year_deg == pytest.approx(0.877, abs=0.002)
    assert rate[0] > 0 and abs(rate[1]) < 1e-15, "push points to a node of 90 deg"
    assert 3074.7 * np.deg2rad(per_year_deg) == pytest.approx(47.1, abs=0.1)
    pole = np.array([0.0, np.deg2rad(7.4)])
    assert np.linalg.norm(natural_inclination_rate_rad_per_day(pole)) < 1e-18


def test_natural_path_is_the_exact_solution_of_the_rate_model() -> None:
    start = np.deg2rad([0.01, -0.02])
    path = natural_inclination_path_rad(start, np.array([0.0, 1e-3, 365.25]))
    assert np.allclose(path[0], start, atol=1e-15)
    rate = (path[1] - path[0]) / 1e-3
    assert np.allclose(rate, natural_inclination_rate_rad_per_day(start), rtol=1e-6)

    from_origin = natural_inclination_path_rad(np.zeros(2), [365.25])[0]
    assert np.rad2deg(np.linalg.norm(from_origin)) == pytest.approx(0.877, abs=0.003)
    assert from_origin[0] > 0, "an uncontrolled plane leaves toward a node of 90 deg"

    pole = np.array([0.0, np.deg2rad(7.4)])
    half_period = natural_inclination_path_rad(start, [53.0 * 365.25 / 2])[0]
    assert np.allclose(half_period - pole, -(start - pole), atol=1e-12)


def test_weekly_north_south_burns_are_found_and_close_the_budget() -> None:
    rng = np.random.default_rng(20)
    t = sample_times(180.0, rng)
    push = natural_push_at_origin()
    period = 7.0
    phase = np.mod(t, period)
    ivec = (phase - period / 2)[:, None] * push[None, :]
    ivec += rng.normal(0.0, 2.5 * ARCSEC, ivec.shape)
    history = geo(rng, t, ivec)

    burns = [m for m in detect_manoeuvres(history) if m.kind == "out_of_plane"]
    expected = int(t[-1] // period)
    assert abs(len(burns) - expected) <= 2
    one_burn = 3074.7 * np.linalg.norm(push) * period
    assert np.mean([m.delta_v_m_s for m in burns]) == pytest.approx(one_burn, rel=0.1)

    analysis = analyse_steps(history, "out_of_plane")
    budget = out_of_plane_budget(history, analysis, burns)
    assert budget.closure == pytest.approx(1.0, abs=0.12)


def test_continuous_control_is_invisible_but_containment_still_measures_it() -> None:
    """The INMARSAT situation: held in place by corrections too small to see."""
    rng = np.random.default_rng(21)
    t = sample_times(180.0, rng)
    ivec = rng.normal(0.0, 2.5 * ARCSEC, (t.size, 2))
    history = geo(rng, t, ivec)

    assert [m for m in detect_manoeuvres(history) if m.kind == "out_of_plane"] == []
    analysis = analyse_steps(history, "out_of_plane")
    budget = out_of_plane_budget(history, analysis, [])
    expected = 3074.7 * np.linalg.norm(natural_push_at_origin()) * (t[-1] - t[0])
    assert budget.required_m_s == pytest.approx(expected, rel=0.05)
    assert budget.closure == 0.0


def test_an_uncontrolled_satellite_requires_no_propellant() -> None:
    rng = np.random.default_rng(22)
    t = sample_times(180.0, rng)
    ivec = np.empty((t.size, 2))
    ivec[0] = np.deg2rad([6.0, 2.0])
    for k in range(1, t.size):
        ivec[k] = ivec[k - 1] + natural_inclination_rate_rad_per_day(ivec[k - 1]) * (
            t[k] - t[k - 1]
        )
    history = geo(rng, t, ivec)
    budget = out_of_plane_budget(history, analyse_steps(history, "out_of_plane"), [])
    assert budget.required_m_s < 0.5


def test_east_west_budget_has_no_requirement() -> None:
    rng = np.random.default_rng(23)
    t = sample_times(60.0, rng)
    history = geo(rng, t, np.zeros((t.size, 2)))
    analysis = analyse_steps(history, "in_plane")
    budget = in_plane_budget(history, analysis, [])
    assert budget.required_m_s is None and budget.closure is None


# --------------------------------------------------------------------------
# End to end from element sets
# --------------------------------------------------------------------------


def test_analyse_from_element_sets_reports_quality_and_the_burn() -> None:
    rng = np.random.default_rng(30)
    t = sample_times(50.0, rng)
    a = 6800.0 - 0.055 * t + rng.normal(0.0, 0.005, t.size)
    a[t > 25.0] += 1.2
    n_rev_day = np.sqrt(MU / a**3) * 86400.0 / (2.0 * np.pi)

    records = []
    for day, n in zip(t, n_rev_day, strict=True):
        line1 = build_line1(epoch_year="26", epoch_day=10.0 + day)
        records.append(parse_tle(line1, build_line2(mean_motion_rev_per_day=float(n))))
    duplicate = build_line1(epoch_year="26", epoch_day=10.0 + t[5] + 1e-8)
    records.append(
        parse_tle(duplicate, build_line2(mean_motion_rev_per_day=float(n_rev_day[5])))
    )

    report = analyse(records)
    assert report.history.quality.refits_merged == 1
    burns = [m for m in report.manoeuvres if m.kind == "in_plane"]
    assert len(burns) == 1
    assert burns[0].delta_a_km == pytest.approx(1.2, abs=0.03)


def test_mixed_objects_are_rejected() -> None:
    a = parse_tle(build_line1(norad_id=1), build_line2(norad_id=1))
    b = parse_tle(build_line1(norad_id=2), build_line2(norad_id=2))
    with pytest.raises(ValueError, match="several objects"):
        analyse([a, b])


# --------------------------------------------------------------------------
# Real data: NASA's announced ISS reboosts
# --------------------------------------------------------------------------

ANNOUNCED_ISS_REBOOSTS = (
    "2025-09-26",
    "2025-10-14",
    "2025-11-07",
    "2025-11-19",
    "2025-12-29",
    "2026-01-23",
    "2026-02-19",
    "2026-03-13",
    "2026-04-16",
    "2026-08-27",
)
"""From NASA's space station blog: Dragon CRS-33 boosts, Progress 93 and the
Cygnus XL boost of 27 August 2026."""


@pytest.mark.network
def test_every_announced_iss_reboost_is_detected() -> None:
    from datetime import date

    from orbwatch.catalog.spacetrack import AuthenticationError, SpaceTrackClient

    try:
        client = SpaceTrackClient.from_env()
    except AuthenticationError as error:
        pytest.skip(f"no Space-Track credentials: {error}")

    report = analyse(
        client.element_set_history(25544, date(2025, 9, 21), date(2026, 9, 21))
    )
    raises = [
        m for m in report.manoeuvres if m.kind == "in_plane" and m.delta_a_km > 0.1
    ]
    for day in ANNOUNCED_ISS_REBOOSTS:
        noon = datetime.fromisoformat(day).replace(tzinfo=UTC) + timedelta(hours=12)
        assert any(
            m.start_utc - timedelta(days=1.5) <= noon <= m.end_utc + timedelta(days=1.5)
            for m in raises
        ), f"announced reboost on {day} not detected"
    assert report.in_plane_budget.closure == pytest.approx(1.0, abs=0.1)
