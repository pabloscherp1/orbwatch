"""Tests for two-line element set parsing.

**Why the offline fixtures are synthetic.** Element sets are redistributed under
terms that this repository is not going to try to interpret, so no real
catalogue data is committed here. The fixtures are built column by column with
correct checksums, which exercises every field and every error path offline.

That leaves one real gap: a fixture I built and a parser I built can share the
same misunderstanding of the format. The check that closes it is
``test_parser_agrees_with_sgp4_on_live_data``, which fetches a current element
set and asserts my parsed fields against the ``sgp4`` library's independent
parser. It is marked ``network`` and is the authoritative validation of this
module. Run it with ``pytest -m network``.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from orbwatch.catalog.tle import (
    TLE_LINE_LENGTH,
    TleFormatError,
    parse_tle,
    parse_tle_text,
    tle_checksum,
)

# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------


def build_line1(
    norad_id: int = 25544,
    classification: str = "U",
    international: str = "98067A  ",
    epoch_year: str = "26",
    epoch_day: float = 258.36891387,
    ndot_half_rev_per_day2: float = 0.00005779,
    nddot_field: str = " 00000+0",
    bstar_field: str = " 11250-3",
    ephemeris_type: int = 0,
    element_set_number: int = 999,
) -> str:
    """Assemble a TLE line 1 from its fixed columns.

    The two drag terms are passed as raw 8-character fields rather than as
    floats, because encoding the implied decimal point is itself parser logic
    and a test should not depend on the thing it is testing.
    """
    ndot_sign = "-" if ndot_half_rev_per_day2 < 0 else " "
    ndot_body = f"{abs(ndot_half_rev_per_day2):.8f}"[1:]
    body = (
        f"1 {norad_id:05d}{classification}"
        f" {international:<8s}"
        f" {epoch_year}{epoch_day:012.8f}"
        f" {ndot_sign}{ndot_body}"
        f" {nddot_field:>8s}"
        f" {bstar_field:>8s}"
        f" {ephemeris_type:1d}"
        f" {element_set_number:4d}"
    )
    assert len(body) == 68, f"line 1 body is {len(body)} chars, expected 68"
    return body + str(tle_checksum(body))


def build_line2(
    norad_id: int = 25544,
    inclination_deg: float = 51.6311,
    raan_deg: float = 213.7631,
    eccentricity_field: str = "0004926",
    argp_deg: float = 142.7188,
    mean_anomaly_deg: float = 217.4143,
    mean_motion_rev_per_day: float = 15.49122235,
    revolution_number: int = 58573,
) -> str:
    """Assemble a TLE line 2 from its fixed columns."""
    body = (
        f"2 {norad_id:05d}"
        f" {inclination_deg:8.4f}"
        f" {raan_deg:8.4f}"
        f" {eccentricity_field:>7s}"
        f" {argp_deg:8.4f}"
        f" {mean_anomaly_deg:8.4f}"
        f" {mean_motion_rev_per_day:11.8f}"
        f"{revolution_number:5d}"
    )
    assert len(body) == 68, f"line 2 body is {len(body)} chars, expected 68"
    return body + str(tle_checksum(body))


# --------------------------------------------------------------------------
# Checksum, tested independently of the fixture builders
# --------------------------------------------------------------------------


def test_checksum_of_all_zeros_is_zero() -> None:
    assert tle_checksum("0" * 68) == 0


def test_checksum_sums_digits_modulo_ten() -> None:
    # Seven ones and sixty-one zeros sums to 7.
    assert tle_checksum("1" * 7 + "0" * 61) == 7
    # Fifteen ones sums to 15, which is 5 modulo 10.
    assert tle_checksum("1" * 15 + "0" * 53) == 5


def test_checksum_counts_a_minus_sign_as_one_and_ignores_other_symbols() -> None:
    assert tle_checksum("-" * 3 + "0" * 65) == 3
    assert tle_checksum("+.  " + "0" * 64) == 0
    assert tle_checksum("-9+.8 " + "0" * 62) == (1 + 9 + 8) % 10


def test_checksum_ignores_the_stated_digit_in_column_69() -> None:
    body = "1" * 68
    assert tle_checksum(body + "0") == tle_checksum(body + "9")


# --------------------------------------------------------------------------
# Field parsing
# --------------------------------------------------------------------------


def test_every_field_parses_from_a_synthetic_element_set() -> None:
    tle = parse_tle(build_line1(), build_line2(), name="ISS (ZARYA)")

    assert tle.norad_id == 25544
    assert tle.classification == "U"
    assert tle.international_designator == "1998-067A"
    assert tle.name == "ISS (ZARYA)"
    assert tle.ephemeris_type == 0
    assert tle.element_set_number == 999
    assert tle.revolution_number == 58573

    assert tle.eccentricity == pytest.approx(0.0004926)
    assert tle.mean_motion_rev_per_day == pytest.approx(15.49122235)
    assert tle.bstar_per_earth_radius == pytest.approx(0.11250e-3)

    assert np.rad2deg(tle.inclination_rad) == pytest.approx(51.6311)
    assert np.rad2deg(tle.raan_rad) == pytest.approx(213.7631)
    assert np.rad2deg(tle.argp_rad) == pytest.approx(142.7188)
    assert np.rad2deg(tle.mean_anomaly_rad) == pytest.approx(217.4143)

    assert len(tle.line1) == TLE_LINE_LENGTH
    assert len(tle.line2) == TLE_LINE_LENGTH


def test_mean_motion_derivatives_are_scaled_out_of_their_tle_encoding() -> None:
    """The format stores n-dot over two and n-double-dot over six.

    Both are rescaled on parse so the attribute name matches the quantity. This
    is the kind of factor that silently poisons a drag analysis if it is missed.
    """
    tle = parse_tle(
        build_line1(ndot_half_rev_per_day2=0.00005779, nddot_field=" 12345-5"),
        build_line2(),
    )
    assert tle.mean_motion_dot_rev_per_day2 == pytest.approx(2.0 * 0.00005779)
    assert tle.mean_motion_ddot_rev_per_day3 == pytest.approx(6.0 * 0.12345e-5)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        (" 11250-3", 0.11250e-3),
        ("-11606-4", -0.11606e-4),
        (" 00000+0", 0.0),
        (" 00000-0", 0.0),
        (" 36258+1", 0.36258e1),
    ],
)
def test_assumed_decimal_drag_fields(field: str, expected: float) -> None:
    """The drag terms omit the decimal point and the exponent marker.

    ``" 11250-3"`` means 0.11250e-3. Reading it as a plain float gives 11250,
    which is wrong by eight orders of magnitude and does not look wrong.
    """
    tle = parse_tle(build_line1(bstar_field=field), build_line2())
    assert tle.bstar_per_earth_radius == pytest.approx(expected)


@pytest.mark.parametrize(
    ("packed", "expected"),
    [
        ("98067A  ", "1998-067A"),
        ("26001AB ", "2026-001AB"),
        ("57001A  ", "1957-001A"),
        ("00001A  ", "2000-001A"),
        ("        ", ""),
    ],
)
def test_international_designator_formatting(packed: str, expected: str) -> None:
    tle = parse_tle(build_line1(international=packed), build_line2())
    assert tle.international_designator == expected


# --------------------------------------------------------------------------
# Epoch
# --------------------------------------------------------------------------


def test_epoch_day_of_year_is_one_based() -> None:
    """Day 1.0 is midnight on 1 January, not midnight on 2 January."""
    tle = parse_tle(build_line1(epoch_year="26", epoch_day=1.0), build_line2())
    assert tle.epoch_utc == datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def test_epoch_fractional_day_resolves_to_sub_second() -> None:
    tle = parse_tle(build_line1(epoch_year="26", epoch_day=1.5), build_line2())
    assert tle.epoch_utc == datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("two_digit", "expected_year"),
    [("57", 1957), ("99", 1999), ("00", 2000), ("26", 2026), ("56", 2056)],
)
def test_two_digit_year_window(two_digit: str, expected_year: int) -> None:
    """57 to 99 mean the twentieth century, 00 to 56 the twenty-first.

    The format genuinely runs out in 2057. This test documents the convention
    rather than endorsing it.
    """
    tle = parse_tle(build_line1(epoch_year=two_digit, epoch_day=1.0), build_line2())
    assert tle.epoch_utc.year == expected_year


def test_epoch_is_timezone_aware_utc() -> None:
    tle = parse_tle(build_line1(), build_line2())
    assert tle.epoch_utc.tzinfo is not None
    assert tle.epoch_utc.utcoffset() == timedelta(0)


def test_age_days_is_signed_and_requires_an_aware_datetime() -> None:
    tle = parse_tle(build_line1(epoch_year="26", epoch_day=1.0), build_line2())

    assert tle.age_days(datetime(2026, 1, 8, tzinfo=UTC)) == pytest.approx(7.0)
    assert tle.age_days(datetime(2025, 12, 31, tzinfo=UTC)) == pytest.approx(-1.0)

    with pytest.raises(ValueError, match="timezone-aware"):
        tle.age_days(datetime(2026, 1, 8))


# --------------------------------------------------------------------------
# Rejection of malformed input
# --------------------------------------------------------------------------


def test_short_line_is_rejected() -> None:
    with pytest.raises(TleFormatError, match="characters"):
        parse_tle(build_line1()[:-1], build_line2())


def test_swapped_lines_are_rejected() -> None:
    with pytest.raises(TleFormatError, match="line number"):
        parse_tle(build_line2(), build_line1())


def test_bad_checksum_is_rejected_by_default_and_can_be_overridden() -> None:
    line1 = build_line1()
    corrupted = line1[:68] + str((int(line1[68]) + 1) % 10)

    with pytest.raises(TleFormatError, match="checksum"):
        parse_tle(corrupted, build_line2())

    tle = parse_tle(corrupted, build_line2(), validate_checksum=False)
    assert tle.norad_id == 25544


def test_lines_from_different_objects_are_rejected() -> None:
    with pytest.raises(TleFormatError, match="do not belong together"):
        parse_tle(build_line1(norad_id=25544), build_line2(norad_id=43013))


def test_trailing_carriage_return_is_tolerated() -> None:
    """Celestrak and Space-Track both serve CRLF."""
    tle = parse_tle(build_line1() + "\r\n", build_line2() + "\r\n")
    assert tle.norad_id == 25544
    assert len(tle.line1) == TLE_LINE_LENGTH


# --------------------------------------------------------------------------
# Multi-record text
# --------------------------------------------------------------------------


def test_parse_text_handles_two_line_three_line_and_blank_lines() -> None:
    text = "\n".join(
        [
            "ISS (ZARYA)",
            build_line1(norad_id=25544),
            build_line2(norad_id=25544),
            "",
            build_line1(norad_id=43013),
            build_line2(norad_id=43013),
            "SOME UPPER STAGE",
            build_line1(norad_id=11111),
            build_line2(norad_id=11111),
        ]
    )
    records = parse_tle_text(text)

    assert [r.norad_id for r in records] == [25544, 43013, 11111]
    assert [r.name for r in records] == ["ISS (ZARYA)", None, "SOME UPPER STAGE"]


def test_parse_text_rejects_a_truncated_record() -> None:
    text = "\n".join(["ISS (ZARYA)", build_line1()])
    with pytest.raises(TleFormatError, match="line 2 missing"):
        parse_tle_text(text)


def test_parse_text_rejects_a_dangling_title() -> None:
    with pytest.raises(TleFormatError, match="no element lines"):
        parse_tle_text("ISS (ZARYA)\n")


# --------------------------------------------------------------------------
# The real validation
# --------------------------------------------------------------------------


@pytest.mark.network
def test_parser_agrees_with_sgp4_on_live_data(tmp_path: Path) -> None:
    """Cross-check every shared field against the sgp4 library's own parser.

    This is the test that actually proves the column map, because sgp4 is an
    independent implementation. The offline fixtures cannot prove it, since I
    wrote both the fixture builder and the parser.
    """
    from sgp4.api import Satrec

    from orbwatch.catalog.sources import fetch_celestrak_tle

    tle = fetch_celestrak_tle(25544, cache_dir=tmp_path, max_cache_age_s=0.0)
    assert tle.norad_id == 25544

    reference = Satrec.twoline2rv(tle.line1, tle.line2)

    assert tle.norad_id == reference.satnum
    assert tle.eccentricity == pytest.approx(reference.ecco, abs=1e-12)
    assert tle.inclination_rad == pytest.approx(reference.inclo, abs=1e-9)
    assert tle.raan_rad == pytest.approx(reference.nodeo, abs=1e-9)
    assert tle.argp_rad == pytest.approx(reference.argpo, abs=1e-9)
    assert tle.mean_anomaly_rad == pytest.approx(reference.mo, abs=1e-9)
    assert tle.bstar_per_earth_radius == pytest.approx(reference.bstar, abs=1e-12)

    # sgp4 carries mean motion in rad/min, this module keeps rev/day.
    rev_per_day = reference.no_kozai * 1440.0 / (2.0 * np.pi)
    assert tle.mean_motion_rev_per_day == pytest.approx(rev_per_day, rel=1e-12)

    # Julian date of the epoch, cross-checked against sgp4's own split value.
    julian_epoch = reference.jdsatepoch + reference.jdsatepochF
    unix_seconds = (julian_epoch - 2440587.5) * 86400.0
    expected_epoch = datetime.fromtimestamp(unix_seconds, tz=UTC)
    assert abs((tle.epoch_utc - expected_epoch).total_seconds()) < 1e-3
