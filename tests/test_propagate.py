"""Tests for the SGP4 propagation boundary.

SGP4 itself is not under test here. It is Vallado's reference implementation
and has its own verification suite. What is under test is everything this
module adds around it: that times reach SGP4 correctly, that errors cannot slip
through as plausible numbers, that stale element sets are flagged, and that the
output cannot be modified after the fact.
"""

import warnings
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sgp4.api import WGS72, Satrec

from orbwatch.catalog.propagate import (
    PropagationError,
    StaleElementSetWarning,
    TemeEphemeris,
    propagate,
    propagate_span,
)
from orbwatch.catalog.tle import TLE, parse_tle
from tests.test_tle import build_line1, build_line2


@pytest.fixture
def iss_like() -> TLE:
    return parse_tle(build_line1(), build_line2(), name="ISS-LIKE")


@pytest.fixture
def decaying() -> TLE:
    """Low perigee and very heavy drag. Fine at epoch, SGP4 error 1 by day 1."""
    return parse_tle(
        build_line1(bstar_field=" 50000-1", ndot_half_rev_per_day2=0.01),
        build_line2(mean_motion_rev_per_day=16.2),
    )


@pytest.fixture
def underground() -> TLE:
    """Mean motion so high the orbit lies below the Earth's surface."""
    return parse_tle(build_line1(), build_line2(mean_motion_rev_per_day=19.0))


# --------------------------------------------------------------------------
# Times reach SGP4 correctly
# --------------------------------------------------------------------------


def test_state_at_epoch_matches_a_direct_sgp4_call(iss_like: TLE) -> None:
    """The wrapper's datetime conversion must land on SGP4's own epoch.

    SGP4 is called directly at its internally parsed epoch, bypassing all of
    this module's time handling. The TLE epoch carries eight decimal places of
    a day and ``datetime`` rounds to the microsecond, so agreement should be
    within about 4 mm at orbital speed. 1 cm is the tolerance.
    """
    ephemeris = propagate(iss_like, iss_like.epoch_utc)

    reference = Satrec.twoline2rv(iss_like.line1, iss_like.line2, WGS72)
    error, r_km, v_km_s = reference.sgp4(reference.jdsatepoch, reference.jdsatepochF)
    assert error == 0

    assert ephemeris.r_teme_km[0] == pytest.approx(r_km, abs=1e-5)
    assert ephemeris.v_teme_km_s[0] == pytest.approx(v_km_s, abs=1e-8)


def test_array_call_matches_individual_calls(iss_like: TLE) -> None:
    offsets_min = [0, 10, 60, 360, 1440]
    times = [iss_like.epoch_utc + timedelta(minutes=m) for m in offsets_min]

    batch = propagate(iss_like, times)
    for k, time in enumerate(times):
        single = propagate(iss_like, time)
        np.testing.assert_allclose(batch.r_teme_km[k], single.r_teme_km[0], atol=1e-9)
        np.testing.assert_allclose(
            batch.v_teme_km_s[k], single.v_teme_km_s[0], atol=1e-12
        )


def test_state_is_physically_plausible_over_a_day(iss_like: TLE) -> None:
    """Loose sanity bounds for a 15.49 rev/day, near-circular orbit.

    The inclination check is the interesting one. The TLE carries a *mean*
    inclination of 51.6311 deg, and the instantaneous inclination computed
    from SGP4's state swings about 0.02 deg either side of it over a day,
    measured on 2026-09-16. That swing is the short-period J2 effect that the
    mean elements average away, and it is exactly why a TLE is not a state
    vector. The tolerance is 0.05 deg.
    """
    times = [iss_like.epoch_utc + timedelta(minutes=m) for m in range(0, 1440, 10)]
    ephemeris = propagate(iss_like, times)

    radius_km = np.linalg.norm(ephemeris.r_teme_km, axis=1)
    speed_km_s = np.linalg.norm(ephemeris.v_teme_km_s, axis=1)
    assert np.all((radius_km > 6750.0) & (radius_km < 6850.0))
    assert np.all((speed_km_s > 7.60) & (speed_km_s < 7.72))

    h = np.cross(ephemeris.r_teme_km, ephemeris.v_teme_km_s)
    inclination_deg = np.rad2deg(np.arccos(h[:, 2] / np.linalg.norm(h, axis=1)))
    mean_inclination_deg = np.rad2deg(iss_like.inclination_rad)
    assert np.all(np.abs(inclination_deg - mean_inclination_deg) < 0.05)


def test_a_single_datetime_gives_a_one_sample_ephemeris(iss_like: TLE) -> None:
    ephemeris = propagate(iss_like, iss_like.epoch_utc)
    assert len(ephemeris) == 1
    assert ephemeris.r_teme_km.shape == (1, 3)
    assert ephemeris.v_teme_km_s.shape == (1, 3)


def test_naive_datetimes_and_empty_input_are_rejected(iss_like: TLE) -> None:
    with pytest.raises(ValueError, match="naive"):
        propagate(iss_like, datetime(2026, 9, 16, 8, 0))
    with pytest.raises(ValueError, match="at least one"):
        propagate(iss_like, [])


# --------------------------------------------------------------------------
# Errors cannot slip through
# --------------------------------------------------------------------------


def test_sgp4_error_raises_by_default_and_names_code_and_time(decaying: TLE) -> None:
    times = [decaying.epoch_utc, decaying.epoch_utc + timedelta(days=1)]
    with pytest.raises(PropagationError, match=r"SGP4 error 1 .* 1 of 2"):
        propagate(decaying, times)


def test_nan_mode_keeps_good_samples_and_blanks_failed_ones(decaying: TLE) -> None:
    times = [decaying.epoch_utc, decaying.epoch_utc + timedelta(days=1)]
    ephemeris = propagate(decaying, times, on_error="nan")

    assert ephemeris.error_codes.tolist() == [0, 1]
    assert ephemeris.valid.tolist() == [True, False]
    assert np.all(np.isfinite(ephemeris.r_teme_km[0]))
    assert np.all(np.isnan(ephemeris.r_teme_km[1]))
    assert np.all(np.isnan(ephemeris.v_teme_km_s[1]))


def test_decayed_orbit_is_caught_even_though_sgp4_returns_finite_numbers(
    underground: TLE,
) -> None:
    """The case that justifies trusting the error code over the numbers.

    For this element set the raw library returns error 6 alongside a position
    about 5,900 km from the Earth's centre: finite, plausible at a glance, and
    underground. A wrapper that only checked for NaN would pass it through.
    """
    raw = Satrec.twoline2rv(underground.line1, underground.line2, WGS72)
    code, r_km, _ = raw.sgp4(raw.jdsatepoch, raw.jdsatepochF)
    assert code == 6
    assert np.all(np.isfinite(r_km)), "premise: the library returns finite numbers"

    with pytest.raises(PropagationError, match="SGP4 error 6"):
        propagate(underground, underground.epoch_utc)

    blanked = propagate(underground, underground.epoch_utc, on_error="nan")
    assert np.all(np.isnan(blanked.r_teme_km[0]))


def test_invalid_on_error_is_rejected(iss_like: TLE) -> None:
    with pytest.raises(ValueError, match="on_error"):
        propagate(iss_like, iss_like.epoch_utc, on_error="ignore")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------


def test_propagating_far_from_epoch_warns(iss_like: TLE) -> None:
    with pytest.warns(StaleElementSetWarning, match="30.0 days"):
        propagate(iss_like, iss_like.epoch_utc + timedelta(days=30))


def test_propagating_backwards_far_from_epoch_also_warns(iss_like: TLE) -> None:
    with pytest.warns(StaleElementSetWarning):
        propagate(iss_like, iss_like.epoch_utc - timedelta(days=30))


def test_propagating_near_epoch_does_not_warn(iss_like: TLE) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", StaleElementSetWarning)
        propagate(iss_like, iss_like.epoch_utc + timedelta(days=1))


def test_ages_are_signed_days_from_epoch(iss_like: TLE) -> None:
    times = [
        iss_like.epoch_utc - timedelta(hours=12),
        iss_like.epoch_utc,
        iss_like.epoch_utc + timedelta(days=2),
    ]
    ephemeris = propagate(iss_like, times)
    np.testing.assert_allclose(ephemeris.ages_days, [-0.5, 0.0, 2.0], atol=1e-9)


# --------------------------------------------------------------------------
# Output integrity
# --------------------------------------------------------------------------


def test_output_arrays_are_read_only(iss_like: TLE) -> None:
    ephemeris = propagate(iss_like, iss_like.epoch_utc)
    with pytest.raises(ValueError, match="read-only"):
        ephemeris.r_teme_km[0, 0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        ephemeris.v_teme_km_s[0, 0] = 0.0


def test_ephemeris_copies_its_inputs_rather_than_freezing_the_callers_arrays() -> None:
    r = np.zeros((1, 3))
    v = np.zeros((1, 3))
    TemeEphemeris(
        norad_id=1,
        tle_epoch_utc=datetime(2026, 9, 16, tzinfo=UTC),
        times_utc=(datetime(2026, 9, 16, tzinfo=UTC),),
        r_teme_km=r,
        v_teme_km_s=v,
        error_codes=np.zeros(1, dtype=int),
    )
    r[0, 0] = 1.0  # must still be writable


def test_ephemeris_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="expected 2 samples"):
        TemeEphemeris(
            norad_id=1,
            tle_epoch_utc=datetime(2026, 9, 16, tzinfo=UTC),
            times_utc=(
                datetime(2026, 9, 16, tzinfo=UTC),
                datetime(2026, 9, 17, tzinfo=UTC),
            ),
            r_teme_km=np.zeros((1, 3)),
            v_teme_km_s=np.zeros((2, 3)),
            error_codes=np.zeros(2, dtype=int),
        )


def test_propagate_span_samples_the_requested_grid(iss_like: TLE) -> None:
    start = iss_like.epoch_utc
    ephemeris = propagate_span(iss_like, start, start + timedelta(hours=1), 60.0)

    assert len(ephemeris) == 61
    assert ephemeris.times_utc[0] == start
    assert ephemeris.times_utc[-1] == start + timedelta(hours=1)


# --------------------------------------------------------------------------
# Live data
# --------------------------------------------------------------------------


@pytest.mark.network
def test_live_iss_propagates_to_a_plausible_state_now(tmp_path) -> None:
    from orbwatch.catalog.sources import fetch_celestrak_tle

    tle = fetch_celestrak_tle(25544, cache_dir=tmp_path)
    ephemeris = propagate(tle, datetime.now(UTC))

    radius_km = float(np.linalg.norm(ephemeris.r_teme_km[0]))
    assert 6600.0 < radius_km < 6900.0
