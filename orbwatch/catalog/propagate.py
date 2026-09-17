"""SGP4 propagation of two-line element sets.

This is the only sanctioned route from a TLE to a Cartesian state in ORBWATCH.
A TLE's elements are mean elements defined inside the SGP4 theory, so they
must be turned into positions by SGP4 itself. Converting them with two-body
formulae instead was measured on 2026-09-16 at about 12 km of position error at
epoch for the ISS, growing to about 470 km one day later.

SGP4 is not reimplemented here. The ``sgp4`` library is Brandon Rhodes' Python
wrapper around Vallado's reference C++ code from "Revisiting Spacetrack Report
#3" (AIAA 2006-6753). What this module adds is the boundary around it:
timezone-safe time handling, error codes turned into exceptions, a warning
when an element set is being used far from its epoch, and output that states
its frame and units.

**Output frame is TEME** (True Equator, Mean Equinox), km and km/s. It is not
J2000 and not GCRF. Convert with :mod:`orbwatch.catalog.frames` before
comparing against anything that is not also TEME.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sgp4.api import SGP4_ERRORS, WGS72, Satrec

from orbwatch.catalog.timescales import (
    SECONDS_PER_DAY,
    julian_date_split_array,
    require_utc,
    time_grid,
)
from orbwatch.catalog.tle import TLE

STALE_ELEMENT_SET_DAYS: float = 14.0
"""Propagating further than this from the epoch triggers a warning.

Skyfield's documentation says an element set "might only be useful for a couple
of weeks to either side of its epoch". This is an upper bound for a quiet
object in LEO, not a guarantee of anything: an object that manoeuvres makes its
element set wrong immediately, regardless of age.
"""


class PropagationError(RuntimeError):
    """Raised when SGP4 reports an error for a requested time."""


class StaleElementSetWarning(UserWarning):
    """Issued when an element set is propagated far from its epoch."""


@dataclass(frozen=True)
class TemeEphemeris:
    """Positions and velocities from SGP4, in the TEME frame.

    Arrays are copied on construction and made read-only, so an ephemeris
    shared between two parts of a pipeline cannot be modified by one of them
    behind the other's back.

    Attributes
    ----------
    norad_id : int
        Catalogue number of the propagated object.
    tle_epoch_utc : datetime
        Epoch of the element set that produced this ephemeris.
    times_utc : tuple of datetime
        Sample instants, timezone-aware UTC, length N.
    r_teme_km : ndarray, shape (N, 3)
        Position in TEME, km. Rows where SGP4 failed are NaN.
    v_teme_km_s : ndarray, shape (N, 3)
        Velocity in TEME, km/s. Rows where SGP4 failed are NaN.
    error_codes : ndarray, shape (N,)
        SGP4 error code per sample, 0 where propagation succeeded.
    """

    norad_id: int
    tle_epoch_utc: datetime
    times_utc: tuple[datetime, ...]
    r_teme_km: NDArray[np.float64]
    v_teme_km_s: NDArray[np.float64]
    error_codes: NDArray[np.int_]

    def __post_init__(self) -> None:
        count = len(self.times_utc)
        r = np.array(self.r_teme_km, dtype=float)
        v = np.array(self.v_teme_km_s, dtype=float)
        codes = np.array(self.error_codes, dtype=int)

        if r.shape != (count, 3) or v.shape != (count, 3) or codes.shape != (count,):
            raise ValueError(
                f"expected {count} samples with shapes ({count}, 3), ({count}, 3) "
                f"and ({count},), got {r.shape}, {v.shape} and {codes.shape}"
            )

        for array in (r, v, codes):
            array.setflags(write=False)

        # A frozen dataclass forbids ordinary attribute assignment, including in
        # __post_init__. object.__setattr__ is the standard way around that for
        # replacing a field with a validated copy during construction.
        object.__setattr__(self, "r_teme_km", r)
        object.__setattr__(self, "v_teme_km_s", v)
        object.__setattr__(self, "error_codes", codes)

    def __len__(self) -> int:
        return len(self.times_utc)

    @property
    def valid(self) -> NDArray[np.bool_]:
        """Boolean mask, True where SGP4 succeeded."""
        return self.error_codes == 0

    @property
    def ages_days(self) -> NDArray[np.float64]:
        """Signed time from the element set epoch to each sample, days."""
        return np.array(
            [
                (time - self.tle_epoch_utc).total_seconds() / SECONDS_PER_DAY
                for time in self.times_utc
            ]
        )


def propagate(
    tle: TLE,
    times_utc: datetime | Sequence[datetime],
    on_error: Literal["raise", "nan"] = "raise",
    stale_after_days: float = STALE_ELEMENT_SET_DAYS,
) -> TemeEphemeris:
    """Propagate an element set with SGP4 to one or more instants.

    Parameters
    ----------
    tle : TLE
        Parsed element set.
    times_utc : datetime or sequence of datetime
        Timezone-aware instants. A single datetime gives a one-sample ephemeris.
    on_error : {"raise", "nan"}, optional
        ``"raise"`` (default) raises on the first failing sample. ``"nan"``
        keeps going, fills failed rows with NaN and records the codes, which is
        what you want for a long span that runs past an object's decay.
    stale_after_days : float, optional
        Warn if any sample lies further than this from the epoch, days.

    Returns
    -------
    TemeEphemeris
        Positions in km and velocities in km/s, in the TEME frame.

    Raises
    ------
    PropagationError
        If SGP4 reports an error and ``on_error`` is ``"raise"``.
    ValueError
        If no times are given, a time is naive, or ``on_error`` is invalid.

    Warns
    -----
    StaleElementSetWarning
        If any sample is further than ``stale_after_days`` from the epoch.

    Notes
    -----
    **Trust the error code, never the numbers.** For an element set whose orbit
    lies below the Earth's surface, SGP4 returns error 6 together with finite,
    plausible-looking positions. Checking only for NaN would let those through.

    **Gravity constants are WGS-72.** The ``sgp4`` library documentation quotes
    Vallado et al. 2006 as specifying WGS-72 as the default, and notes that
    element sets across the industry are most likely generated with WGS-72.
    Switching to the more modern WGS-84 therefore makes results *worse*, for
    the same reason that treating mean elements as osculating does: the
    elements only mean something under the model and constants they were
    fitted with.
    """
    if on_error not in ("raise", "nan"):
        raise ValueError(f"on_error must be 'raise' or 'nan', got {on_error!r}")

    if isinstance(times_utc, datetime):
        times = (require_utc(times_utc),)
    else:
        times = tuple(require_utc(time) for time in times_utc)
    if not times:
        raise ValueError("at least one time is required")

    satellite = Satrec.twoline2rv(tle.line1, tle.line2, WGS72)
    jd_midnight, day_fraction = julian_date_split_array(times)
    codes, r_km, v_km_s = satellite.sgp4_array(jd_midnight, day_fraction)

    codes = np.asarray(codes, dtype=int)
    r_km = np.array(r_km, dtype=float)
    v_km_s = np.array(v_km_s, dtype=float)

    failed = codes != 0
    if failed.any():
        if on_error == "raise":
            first = int(np.flatnonzero(failed)[0])
            code = int(codes[first])
            meaning = SGP4_ERRORS.get(code, "unknown error")
            raise PropagationError(
                f"SGP4 error {code} for NORAD {tle.norad_id} at "
                f"{times[first].isoformat()}: {meaning}. "
                f"{int(failed.sum())} of {len(times)} requested times failed. "
                "Pass on_error='nan' to keep the samples that succeeded."
            )
        r_km[failed] = np.nan
        v_km_s[failed] = np.nan

    ephemeris = TemeEphemeris(
        norad_id=tle.norad_id,
        tle_epoch_utc=tle.epoch_utc,
        times_utc=times,
        r_teme_km=r_km,
        v_teme_km_s=v_km_s,
        error_codes=codes,
    )

    oldest_days = float(np.max(np.abs(ephemeris.ages_days)))
    if oldest_days > stale_after_days:
        warnings.warn(
            StaleElementSetWarning(
                f"NORAD {tle.norad_id} propagated {oldest_days:.1f} days from its "
                f"element set epoch {tle.epoch_utc.isoformat()}. Accuracy degrades "
                "quickly with age, and any manoeuvre since the epoch is invisible."
            ),
            stacklevel=2,
        )

    return ephemeris


def propagate_span(
    tle: TLE,
    start_utc: datetime,
    stop_utc: datetime,
    step_s: float,
    on_error: Literal["raise", "nan"] = "raise",
    stale_after_days: float = STALE_ELEMENT_SET_DAYS,
) -> TemeEphemeris:
    """Propagate over an evenly spaced span of time.

    Convenience wrapper around :func:`propagate` and
    :func:`orbwatch.catalog.timescales.time_grid`. Parameters and behaviour are
    the same; ``stop_utc`` is included only if it falls exactly on the grid.
    """
    times = time_grid(start_utc, stop_utc, step_s)
    return propagate(
        tle, times, on_error=on_error, stale_after_days=stale_after_days
    )
