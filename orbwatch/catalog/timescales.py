"""Time handling: timezone-aware UTC datetimes and split Julian dates.

Every public function in ORBWATCH takes time as a timezone-aware ``datetime``.
Naive datetimes are rejected everywhere, because a naive datetime is silently
interpreted in the machine's local timezone, and an orbit propagated from a
laptop in Zurich one hour off is several thousand kilometres off in LEO.

**Split Julian dates.** A Julian date near 2.46 million days stored as one
64-bit float has a resolution of about 40 microseconds, which at orbital speed
is about 30 cm. Storing it as a whole part and a fractional part keeps the
fraction at full precision. This module follows the convention of the ``sgp4``
library and of Vallado's reference code: the whole part is the Julian date of
the preceding midnight, so it always ends in ``.5``, and the fraction is the
elapsed fraction of that UTC day, in [0, 1).

**This is UTC and only UTC.** Julian dates here are a continuous count of UTC
calendar days. Python's ``datetime`` has no leap seconds, so an interval that
spans one is short by a second. SGP4 expects exactly this UTC day count, so it
is correct for propagation. It is not correct for anything that needs TT or
UT1, such as a precise Earth rotation angle or the J2000.0 epoch, which is
defined in TT. See :mod:`orbwatch.catalog.frames`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

UNIX_EPOCH_UTC: datetime = datetime(1970, 1, 1, tzinfo=UTC)

UNIX_EPOCH_JULIAN_DATE: float = 2440587.5
"""Julian date of 1970-01-01T00:00:00 UTC. Textbook constant."""

SECONDS_PER_DAY: float = 86400.0


def require_utc(time: datetime) -> datetime:
    """Reject naive datetimes and normalise aware ones to UTC.

    Parameters
    ----------
    time : datetime
        Must be timezone-aware. Any timezone is accepted and converted.

    Returns
    -------
    datetime
        The same instant, expressed in UTC.

    Raises
    ------
    ValueError
        If ``time`` is naive.
    """
    if time.tzinfo is None or time.utcoffset() is None:
        raise ValueError(
            f"datetime {time.isoformat()} is naive. Pass a timezone-aware "
            "datetime, for example datetime(2026, 9, 16, tzinfo=UTC)."
        )
    return time.astimezone(UTC)


def julian_date_split(time_utc: datetime) -> tuple[float, float]:
    """Split Julian date of one instant.

    Parameters
    ----------
    time_utc : datetime
        Timezone-aware datetime.

    Returns
    -------
    jd_midnight : float
        Julian date of the preceding UTC midnight, always ending in ``.5``.
    day_fraction : float
        Elapsed fraction of the UTC day, in [0, 1).

    Notes
    -----
    Built from integer day, second and microsecond counts rather than from a
    float number of seconds, so the fraction is exact to the microsecond.
    ``timedelta`` normalises negative intervals so that seconds are always
    non-negative, which keeps this correct for dates before 1970 too.
    """
    delta = require_utc(time_utc) - UNIX_EPOCH_UTC
    jd_midnight = UNIX_EPOCH_JULIAN_DATE + delta.days
    day_fraction = (delta.seconds + delta.microseconds / 1e6) / SECONDS_PER_DAY
    return jd_midnight, day_fraction


def julian_date_split_array(
    times_utc: Sequence[datetime],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split Julian dates for a sequence of instants.

    Parameters
    ----------
    times_utc : sequence of datetime
        Timezone-aware datetimes.

    Returns
    -------
    jd_midnight : ndarray, shape (N,)
    day_fraction : ndarray, shape (N,)
    """
    pairs = [julian_date_split(time) for time in times_utc]
    jd_midnight = np.array([pair[0] for pair in pairs], dtype=float)
    day_fraction = np.array([pair[1] for pair in pairs], dtype=float)
    return jd_midnight, day_fraction


def time_grid(
    start_utc: datetime,
    stop_utc: datetime,
    step_s: float,
) -> tuple[datetime, ...]:
    """Evenly spaced instants from start to stop.

    Parameters
    ----------
    start_utc, stop_utc : datetime
        Timezone-aware bounds. ``stop_utc`` is included only if it falls
        exactly on the grid.
    step_s : float
        Spacing, seconds. Must be positive.

    Returns
    -------
    tuple of datetime
        UTC instants, starting at ``start_utc``.

    Notes
    -----
    Each instant is computed as ``start + k * step`` rather than by repeatedly
    adding ``step`` to the previous one, so rounding does not accumulate over a
    long grid.
    """
    if step_s <= 0.0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    start = require_utc(start_utc)
    stop = require_utc(stop_utc)
    if stop < start:
        raise ValueError("stop_utc is before start_utc")

    span_s = (stop - start).total_seconds()
    count = int(np.floor(span_s / step_s + 1e-9)) + 1
    return tuple(start + timedelta(seconds=k * step_s) for k in range(count))
