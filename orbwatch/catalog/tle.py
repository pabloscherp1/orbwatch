"""Two-line element set parsing.

Pure parsing and validation. Nothing in this module touches the network, so it
is fully testable offline. Fetching lives in ``orbwatch.catalog.sources``.

A TLE is a fixed-column ASCII format from the 1960s, and the columns are
load-bearing. A field is defined by where it sits, not by the whitespace around
it, so this module slices by index rather than splitting on spaces. Splitting on
whitespace appears to work and then silently mis-parses the first object with an
empty field or a negative exponent.

**A TLE is not a state vector.** The elements carried here are *mean* elements in
a specific analytical theory (Brouwer, with Kozai's mean motion convention) and
they only mean anything when fed back into SGP4, which is the model they were
fitted to. Converting them directly to a Cartesian state with two-body formulae
is wrong by kilometres, and this module deliberately gives you no helper that
would let you do it by accident. Use ``orbwatch.catalog.propagate`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

TLE_LINE_LENGTH: int = 69
"""Every TLE line is exactly 69 characters, excluding the line terminator."""


class TleFormatError(ValueError):
    """Raised when a line is not a structurally valid TLE line."""


def tle_checksum(line: str) -> int:
    """Modulo-10 checksum over the first 68 characters of a TLE line.

    Parameters
    ----------
    line : str
        A TLE line, at least 68 characters.

    Returns
    -------
    int
        The checksum digit, 0 to 9.

    Notes
    -----
    Digits count as their value, a minus sign counts as 1, and everything else
    counts as zero. Plus signs, decimal points and spaces contribute nothing,
    which is why a minus sign is the only non-digit that matters.
    """
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return total % 10


def _parse_assumed_decimal(field: str) -> float:
    """Parse a TLE field with an implied leading decimal point.

    The drag terms are packed as a five-digit mantissa with the decimal point
    assumed in front of it, followed by a signed single-digit exponent, all
    without an ``e``. So ``" 11250-3"`` means 0.11250e-3 and ``"-11606-4"``
    means -0.11606e-4. An all-zero field means exactly zero.
    """
    text = field.strip()
    if not text or set(text) <= {"0", "+", "-"}:
        return 0.0

    sign = 1.0
    if text[0] in "+-":
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:]

    exponent = 0
    mantissa = text
    for index in range(len(text) - 1, 0, -1):
        if text[index] in "+-":
            mantissa = text[:index]
            exponent = int(text[index] + text[index + 1 :])
            break

    return sign * float(f"0.{mantissa}") * 10.0**exponent


def _parse_epoch(two_digit_year: str, day_of_year: str) -> datetime:
    """Convert the TLE epoch fields to a timezone-aware UTC datetime.

    The year is two digits, so it is ambiguous by construction. The standing
    convention, and the one SGP4 implementations use, is that 57 to 99 mean
    1957 to 1999 and 00 to 56 mean 2000 to 2056. The format therefore breaks in
    2057, which is a real limitation rather than a joke.

    The day of year is one-based, so 1.0 is midnight on 1 January.
    """
    year = int(two_digit_year)
    full_year = 1900 + year if year >= 57 else 2000 + year
    return datetime(full_year, 1, 1, tzinfo=UTC) + timedelta(
        days=float(day_of_year) - 1.0
    )


def _format_international_designator(field: str) -> str:
    """Turn the packed launch designator into the conventional form.

    ``"98067A  "`` becomes ``"1998-067A"``. Objects catalogued without a launch
    designator carry a blank field and get an empty string back.
    """
    text = field.rstrip()
    if not text:
        return ""
    year = int(text[:2])
    full_year = 1900 + year if year >= 57 else 2000 + year
    return f"{full_year}-{text[2:5]}{text[5:].strip()}"


@dataclass(frozen=True)
class TLE:
    """One parsed two-line element set.

    Angles are radians and the epoch is timezone-aware UTC. The mean motion and
    its derivatives keep their native TLE units, revolutions per day, because
    converting them to SI here would invite exactly the misuse the module
    docstring warns about: these are fitted coefficients of an analytical
    theory, not osculating quantities, and they belong to SGP4.

    The original ``line1`` and ``line2`` are retained verbatim, because SGP4 is
    fed the raw lines rather than the parsed fields.

    Attributes
    ----------
    norad_id : int
        NORAD catalogue number, the object's identity in the public catalogue.
    classification : str
        ``U`` unclassified, ``C`` classified, ``S`` secret. Public sources are
        always ``U``.
    international_designator : str
        COSPAR ID, for example ``"1998-067A"``. Empty if the object has none.
    epoch_utc : datetime
        Epoch of the element set, timezone-aware UTC. The single most important
        field, because a TLE is only as good as how old it is.
    mean_motion_rev_per_day : float
        Kozai mean motion at epoch, revolutions per day.
    mean_motion_dot_rev_per_day2 : float
        First derivative of mean motion, rev/day^2. The TLE stores half this
        value; it is doubled here so the name matches the quantity.
    mean_motion_ddot_rev_per_day3 : float
        Second derivative of mean motion, rev/day^3. The TLE stores a sixth of
        this value; it is multiplied by six here for the same reason.
    bstar_per_earth_radius : float
        SGP4 drag-like coefficient, inverse Earth radii. Not a ballistic
        coefficient and not physical on its own, it is a fitted parameter.
    ephemeris_type : int
        Almost always 0, meaning the distributed SGP4 model.
    element_set_number : int
        Increments each time a new element set is issued for the object.
    inclination_rad, raan_rad, argp_rad, mean_anomaly_rad : float
        Mean angular elements at epoch, radians.
    eccentricity : float
        Mean eccentricity, unitless.
    revolution_number : int
        Revolution count at epoch. Known to be unreliable, it wraps at 99999
        and is frequently wrong by a revolution.
    line1, line2 : str
        The original lines, verbatim, 69 characters each.
    name : str or None
        Object name from the optional title line, if the source provided one.
    """

    norad_id: int
    classification: str
    international_designator: str
    epoch_utc: datetime
    mean_motion_rev_per_day: float
    mean_motion_dot_rev_per_day2: float
    mean_motion_ddot_rev_per_day3: float
    bstar_per_earth_radius: float
    ephemeris_type: int
    element_set_number: int
    inclination_rad: float
    raan_rad: float
    eccentricity: float
    argp_rad: float
    mean_anomaly_rad: float
    revolution_number: int
    line1: str
    line2: str
    name: str | None = None

    def age_days(self, at_utc: datetime) -> float:
        """Age of this element set at a given time, in days.

        Negative if ``at_utc`` precedes the epoch, which is legitimate: SGP4
        propagates backwards, and a fresh element set is often issued with an
        epoch slightly in the future.

        SGP4 accuracy degrades roughly as the square of this number. Past about
        a week in low Earth orbit the along-track error is typically kilometres,
        and the object may well have manoeuvred since.
        """
        if at_utc.tzinfo is None:
            raise ValueError("at_utc must be timezone-aware")
        return (at_utc - self.epoch_utc).total_seconds() / 86400.0


def _validate_line(line: str, expected_number: str, validate_checksum: bool) -> str:
    stripped = line.rstrip("\r\n")
    if len(stripped) != TLE_LINE_LENGTH:
        raise TleFormatError(
            f"line {expected_number} is {len(stripped)} characters, "
            f"expected {TLE_LINE_LENGTH}: {stripped!r}"
        )
    if stripped[0] != expected_number:
        raise TleFormatError(
            f"expected line number {expected_number}, got {stripped[0]!r}"
        )
    if validate_checksum:
        computed = tle_checksum(stripped)
        stated = stripped[68]
        if not stated.isdigit() or int(stated) != computed:
            raise TleFormatError(
                f"line {expected_number} checksum is {stated!r}, computed {computed}. "
                "The line is corrupt or was edited by hand."
            )
    return stripped


def parse_tle(
    line1: str,
    line2: str,
    name: str | None = None,
    validate_checksum: bool = True,
) -> TLE:
    """Parse a two-line element set.

    Parameters
    ----------
    line1, line2 : str
        The two element lines, 69 characters each. Trailing newline or carriage
        return is tolerated, since most sources serve CRLF.
    name : str or None, optional
        Object name from the title line, if the source supplied one.
    validate_checksum : bool, optional
        Reject lines whose modulo-10 checksum does not match. Default True.
        Turn it off only for a source known to emit bad checksums, and say so
        where you turn it off.

    Returns
    -------
    TLE

    Raises
    ------
    TleFormatError
        If either line is the wrong length, carries the wrong line number, fails
        its checksum, or if the two lines describe different catalogue objects.
    """
    first = _validate_line(line1, "1", validate_checksum)
    second = _validate_line(line2, "2", validate_checksum)

    norad_id = int(first[2:7])
    if int(second[2:7]) != norad_id:
        raise TleFormatError(
            f"line 1 is object {norad_id} but line 2 is object {int(second[2:7])}, "
            "the two lines do not belong together"
        )

    return TLE(
        norad_id=norad_id,
        classification=first[7],
        international_designator=_format_international_designator(first[9:17]),
        epoch_utc=_parse_epoch(first[18:20], first[20:32]),
        mean_motion_rev_per_day=float(second[52:63]),
        mean_motion_dot_rev_per_day2=2.0 * float(first[33:43]),
        mean_motion_ddot_rev_per_day3=6.0 * _parse_assumed_decimal(first[44:52]),
        bstar_per_earth_radius=_parse_assumed_decimal(first[53:61]),
        ephemeris_type=int(first[62]),
        element_set_number=int(first[64:68]),
        inclination_rad=float(np.deg2rad(float(second[8:16]))),
        raan_rad=float(np.deg2rad(float(second[17:25]))),
        eccentricity=float(f"0.{second[26:33]}"),
        argp_rad=float(np.deg2rad(float(second[34:42]))),
        mean_anomaly_rad=float(np.deg2rad(float(second[43:51]))),
        revolution_number=int(second[63:68]),
        line1=first,
        line2=second,
        name=name,
    )


def parse_tle_text(text: str, validate_checksum: bool = True) -> list[TLE]:
    """Parse a text blob containing one or more element sets.

    Handles both the two-line format and the three-line format with a title
    line, mixed freely, which is what Celestrak and Space-Track actually serve.
    Blank lines are skipped.

    Parameters
    ----------
    text : str
        Raw response body or file contents.
    validate_checksum : bool, optional
        Passed through to :func:`parse_tle`.

    Returns
    -------
    list of TLE
        In the order they appeared.

    Raises
    ------
    TleFormatError
        If the file ends mid-record, or a line appears where no line 1 or title
        was expected.
    """
    lines = [line.rstrip("\r\n") for line in text.splitlines()]
    lines = [line for line in lines if line.strip()]

    records: list[TLE] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        name: str | None = None

        if not line.startswith("1 "):
            name = line.strip()
            index += 1
            if index >= len(lines):
                raise TleFormatError(
                    f"text ends after title line {name!r} with no element lines"
                )

        if index + 1 >= len(lines):
            raise TleFormatError(
                f"text ends after line 1 of object at index {index}, line 2 missing"
            )

        records.append(
            parse_tle(
                lines[index],
                lines[index + 1],
                name=name,
                validate_checksum=validate_checksum,
            )
        )
        index += 2

    return records
