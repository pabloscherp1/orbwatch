"""Tests for UTC datetime and split Julian date handling.

The reference values are textbook day counts, and the ``sgp4`` library's own
``jday`` routine, a translation of Vallado's C++ reference code, is used as an
independent implementation to agree with.
"""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from sgp4.api import jday

from orbwatch.catalog.timescales import (
    julian_date_split,
    julian_date_split_array,
    require_utc,
    time_grid,
)


def test_unix_epoch_is_julian_date_2440587_5() -> None:
    jd, fraction = julian_date_split(datetime(1970, 1, 1, tzinfo=UTC))
    assert jd == 2440587.5
    assert fraction == 0.0


def test_noon_on_1_january_2000_utc_is_julian_date_2451545() -> None:
    """A calendar day count, deliberately not labelled "J2000.0".

    The J2000.0 epoch is JD 2451545.0 in TT, which is about a minute earlier
    than noon UTC. Mixing those two up is exactly the kind of time-scale error
    this module's docstring warns about.
    """
    jd, fraction = julian_date_split(datetime(2000, 1, 1, 12, tzinfo=UTC))
    assert jd + fraction == 2451545.0
    assert fraction == 0.5


@pytest.mark.parametrize(
    "instant",
    [
        datetime(2026, 9, 16, 8, 51, 14, 123456, tzinfo=UTC),
        datetime(2000, 2, 29, 23, 59, 59, 999999, tzinfo=UTC),
        datetime(1998, 11, 20, 6, 40, tzinfo=UTC),
        datetime(1965, 7, 4, 17, 3, 30, tzinfo=UTC),
    ],
)
def test_agrees_with_the_sgp4_reference_jday(instant: datetime) -> None:
    jd, fraction = julian_date_split(instant)
    seconds = instant.second + instant.microsecond / 1e6
    ref_jd, ref_fraction = jday(
        instant.year, instant.month, instant.day, instant.hour, instant.minute, seconds
    )
    assert jd == ref_jd
    assert fraction == pytest.approx(ref_fraction, abs=1e-15)


def test_whole_part_ends_in_half_and_fraction_is_in_unit_interval() -> None:
    for day_offset in range(0, 800, 37):
        instant = datetime(1969, 12, 1, 23, 59, tzinfo=UTC) + timedelta(days=day_offset)
        jd, fraction = julian_date_split(instant)
        assert jd % 1.0 == 0.5
        assert 0.0 <= fraction < 1.0


def test_other_timezones_are_converted_not_rejected() -> None:
    zurich_summer = timezone(timedelta(hours=2))
    local = datetime(2026, 9, 16, 10, 0, tzinfo=zurich_summer)
    utc = datetime(2026, 9, 16, 8, 0, tzinfo=UTC)

    assert julian_date_split(local) == julian_date_split(utc)
    assert require_utc(local) == utc
    assert require_utc(local).utcoffset() == timedelta(0)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValueError, match="naive"):
        julian_date_split(datetime(2026, 9, 16, 8, 0))


def test_array_form_matches_scalar_form() -> None:
    instants = [datetime(2026, 9, 16, h, tzinfo=UTC) for h in (0, 6, 12, 18)]
    jd, fraction = julian_date_split_array(instants)
    for k, instant in enumerate(instants):
        assert (jd[k], fraction[k]) == julian_date_split(instant)


def test_time_grid_includes_start_and_stop_when_on_grid() -> None:
    start = datetime(2026, 9, 16, 0, 0, tzinfo=UTC)
    grid = time_grid(start, start + timedelta(minutes=10), step_s=60.0)

    assert len(grid) == 11
    assert grid[0] == start
    assert grid[-1] == start + timedelta(minutes=10)
    spacings = [later - earlier for earlier, later in zip(grid, grid[1:], strict=False)]
    assert all(spacing == timedelta(seconds=60) for spacing in spacings)


def test_time_grid_excludes_stop_when_off_grid() -> None:
    start = datetime(2026, 9, 16, 0, 0, tzinfo=UTC)
    grid = time_grid(start, start + timedelta(seconds=150), step_s=60.0)

    assert len(grid) == 3
    assert grid[-1] == start + timedelta(seconds=120)


def test_time_grid_does_not_accumulate_rounding_over_a_long_span() -> None:
    """Thirty days at a non-integer step. Built as start + k*step, the last
    sample should land exactly where arithmetic says, not drift."""
    start = datetime(2026, 9, 16, tzinfo=UTC)
    step_s = 0.1
    grid = time_grid(start, start + timedelta(hours=1), step_s)

    assert len(grid) == 36001
    assert grid[-1] == start + timedelta(hours=1)


def test_time_grid_rejects_bad_arguments() -> None:
    start = datetime(2026, 9, 16, tzinfo=UTC)
    with pytest.raises(ValueError, match="positive"):
        time_grid(start, start + timedelta(minutes=1), step_s=0.0)
    with pytest.raises(ValueError, match="before"):
        time_grid(start, start - timedelta(minutes=1), step_s=10.0)
