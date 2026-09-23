"""Orbit histories: a pile of element sets turned into clean physical time series.

Raw element set histories are not clean measurements. Three problems were found
the hard way on real data, and each has a step here:

1. **Re-fits published at the same epoch.** Space-Track often carries several
   fits whose epochs differ only in the microsecond digits. Treated as separate
   samples they create gaps of a fraction of a second, which turns any rate into
   noise. They are merged.
2. **Transient element sets.** Some fits land away from the object's path and
   the path then resumes where it was, sometimes for one fit and sometimes for
   runs lasting days. A real manoeuvre changes the orbit and it stays changed; a
   transient does not. They are rejected by a two-sided test, and every
   rejection is reported rather than silently dropped.
3. **The wrong coordinates.** The length of the inclination vector hides what a
   controlled geostationary satellite is doing, so the components are used.

Units follow the names: days for time, km for semi-major axis, radians for
inclination.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from orbwatch.catalog.tle import TLE

MU_WGS72_KM3_S2: float = 398600.8
"""Gravitational parameter SGP4 element sets are fitted with, km^3/s^2."""

DEFAULT_MERGE_WITHIN_S: float = 60.0
DEFAULT_SIDE_WINDOW_DAYS: float = 10.0
DEFAULT_TRANSIENT_SIGMAS: float = 6.0
MIN_GAP_DAYS: float = 1.0 / 24.0
"""Gaps shorter than an hour are never used to estimate rates or steps."""

GEOSYNCHRONOUS_BAND_REV_PER_DAY: tuple[float, float] = (0.9, 1.1)

Regime = Literal["geosynchronous", "other"]


def robust_sigma(values: ArrayLike) -> float:
    """Median absolute deviation scaled to match a Gaussian standard deviation.

    Robust to the very outliers the caller is usually trying to find, which is
    the point: a burn or a bad fit must not inflate the estimate of the noise it
    is being compared against.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def merge_refits(
    records: Sequence[TLE], within_s: float = DEFAULT_MERGE_WITHIN_S
) -> list[TLE]:
    """Collapse element sets whose epochs are closer than ``within_s`` seconds.

    Keeps the last record of each cluster in epoch order. The TLE format carries
    no creation date, so there is no better tie-break; the JSON form of
    Space-Track's history does, and would allow keeping the newest fit instead.
    """
    merged: list[TLE] = []
    for record in sorted(records, key=lambda r: r.epoch_utc):
        if (
            merged
            and (record.epoch_utc - merged[-1].epoch_utc).total_seconds() < within_s
        ):
            merged[-1] = record
        else:
            merged.append(record)
    return merged


def semi_major_axis_km(record: TLE, mu_km3_s2: float = MU_WGS72_KM3_S2) -> float:
    """Semi-major axis implied by the element set's mean motion, km.

    Uses the Kozai mean motion as published, without SGP4's conversion to
    Brouwer's convention. That leaves a near-constant offset per object, a few
    hundred metres in low orbit, which cancels in the step between consecutive
    element sets, and steps are all the event detectors use.
    """
    n_rad_s = record.mean_motion_rev_per_day * 2.0 * np.pi / 86400.0
    return float((mu_km3_s2 / n_rad_s**2) ** (1.0 / 3.0))


def inclination_vector_rad(record: TLE) -> tuple[float, float]:
    """Inclination vector (i sin RAAN, i cos RAAN), radians.

    Smooth through zero inclination, where the node is undefined, which is
    exactly where controlled geostationary satellites live.
    """
    i = record.inclination_rad
    return float(i * np.sin(record.raan_rad)), float(i * np.cos(record.raan_rad))


def two_sided_transients(
    t_days: ArrayLike,
    signal: ArrayLike,
    side_days: float = DEFAULT_SIDE_WINDOW_DAYS,
    sigmas: float = DEFAULT_TRANSIENT_SIGMAS,
    min_side: int = 3,
) -> NDArray[np.bool_]:
    """Flag samples that disagree with the path both before and after them.

    Parameters
    ----------
    t_days : array_like, shape (N,)
        Strictly increasing sample times, days.
    signal : array_like, shape (N,) or (N, M)
        The quantity to test, one row per sample.
    side_days : float, optional
        Length of the window on each side, days.
    sigmas : float, optional
        Rejection threshold in robust standard deviations of the distances.
    min_side : int, optional
        A side with fewer samples than this cannot be judged.

    Returns
    -------
    ndarray of bool, shape (N,)
        True where the sample is transient.

    Notes
    -----
    For each sample, the path on each side is summarised by the median of that
    window, extrapolated to the sample's time with the series' robust overall
    rate so that a steady drift such as drag decay is not mistaken for an
    offset. The sample's distance to the path is the smaller of the two
    distances. A genuine manoeuvre agrees with the path after it, so it is kept.
    A single bad fit agrees with neither side, and a bad run shorter than about
    half the window is out-voted on both sides.

    A centred window, the obvious alternative, fails in both directions on real
    data: after a burn the catalogue can go quiet for many hours, so a centred
    median sits on the old orbit and rejects the correct new points, while a
    bad run of several days out-votes it and is kept.

    A sample whose window on either side is too sparse to judge is kept.

    The threshold errs on the side of rejecting. Six robust sigmas of the
    distances corresponds to roughly four standard deviations of the underlying
    noise, so on clean Gaussian data about one good sample in 1,600 is rejected
    (measured over 200 synthetic years). That trade is deliberate: losing a good
    sample costs nothing, because a genuine step is carried by the next one,
    while keeping a bad one costs two phantom burns.
    """
    t = np.asarray(t_days, dtype=float)
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = t.size
    if n < 2 * min_side + 1:
        return np.zeros(n, dtype=bool)

    dt = np.diff(t)
    usable = dt > MIN_GAP_DAYS
    rate = np.array(
        [
            np.median(np.diff(x[:, c])[usable] / dt[usable]) if usable.any() else 0.0
            for c in range(x.shape[1])
        ]
    )

    before_start = np.searchsorted(t, t - side_days, side="left")
    after_stop = np.searchsorted(t, t + side_days, side="right")

    distance = np.full(n, np.nan)
    for j in range(n):
        lo, hi = before_start[j], j
        a_lo, a_hi = j + 1, after_stop[j]
        if hi - lo < min_side or a_hi - a_lo < min_side:
            continue
        pred_before = np.median(x[lo:hi], axis=0) + rate * (t[j] - np.median(t[lo:hi]))
        pred_after = np.median(x[a_lo:a_hi], axis=0) + rate * (
            t[j] - np.median(t[a_lo:a_hi])
        )
        distance[j] = min(
            np.linalg.norm(x[j] - pred_before), np.linalg.norm(x[j] - pred_after)
        )

    judged = np.isfinite(distance)
    if judged.sum() < 3:
        return np.zeros(n, dtype=bool)
    threshold = np.median(distance[judged]) + sigmas * robust_sigma(distance[judged])
    return judged & (distance > threshold)


@dataclass(frozen=True)
class DataQuality:
    """What cleaning did to a history, so nothing disappears silently."""

    records_in: int
    after_merging_refits: int
    transients_rejected: int
    transient_epochs_utc: tuple[datetime, ...]
    merge_within_s: float

    @property
    def refits_merged(self) -> int:
        return self.records_in - self.after_merging_refits


@dataclass(frozen=True)
class OrbitHistory:
    """Clean time series for one object, ready for event detection.

    Attributes
    ----------
    norad_id : int
    epochs_utc : tuple of datetime
        One per sample, strictly increasing.
    t_days : ndarray, shape (N,)
        Days since the first epoch.
    a_km : ndarray, shape (N,)
        Semi-major axis from mean motion, km.
    inclination_rad : ndarray, shape (N,)
    inclination_vector_rad : ndarray, shape (N, 2)
        Components (i sin RAAN, i cos RAAN), radians.
    regime : {"geosynchronous", "other"}
        Decides which out-of-plane signal is meaningful.
    quality : DataQuality
    mu_km3_s2 : float
    """

    norad_id: int
    epochs_utc: tuple[datetime, ...]
    t_days: NDArray[np.float64]
    a_km: NDArray[np.float64]
    inclination_rad: NDArray[np.float64]
    inclination_vector_rad: NDArray[np.float64]
    regime: Regime
    quality: DataQuality
    mu_km3_s2: float = MU_WGS72_KM3_S2

    def __len__(self) -> int:
        return len(self.epochs_utc)

    @property
    def span_days(self) -> float:
        return float(self.t_days[-1] - self.t_days[0]) if len(self) else 0.0

    @property
    def mean_speed_km_s(self) -> float:
        """Circular speed at the mean semi-major axis, km/s."""
        return float(np.sqrt(self.mu_km3_s2 / np.mean(self.a_km)))

    @property
    def out_of_plane_signal(self) -> NDArray[np.float64]:
        """The inclination vector at GEO, the inclination itself elsewhere.

        In low orbit the node precesses by degrees per day under J2, so the
        inclination vector rotates quickly and a step in it means nothing. The
        inclination alone has no secular J2 drift, so it is the useful signal.
        """
        if self.regime == "geosynchronous":
            return self.inclination_vector_rad
        return self.inclination_rad[:, None]


def classify_regime(records: Sequence[TLE]) -> Regime:
    n = np.median([r.mean_motion_rev_per_day for r in records])
    low, high = GEOSYNCHRONOUS_BAND_REV_PER_DAY
    return "geosynchronous" if low < n < high else "other"


def build_history(
    records: Sequence[TLE],
    merge_within_s: float = DEFAULT_MERGE_WITHIN_S,
    reject_transients: bool = True,
    side_days: float = DEFAULT_SIDE_WINDOW_DAYS,
    transient_sigmas: float = DEFAULT_TRANSIENT_SIGMAS,
) -> OrbitHistory:
    """Clean an element set history and convert it to physical series.

    Parameters
    ----------
    records : sequence of TLE
        Element sets for one object, in any order.
    merge_within_s : float, optional
        Epochs closer than this are treated as re-fits of one epoch.
    reject_transients : bool, optional
        Apply the two-sided transient test to both the semi-major axis and the
        out-of-plane signal, rejecting a sample flagged by either.
    side_days, transient_sigmas : float, optional
        Passed to :func:`two_sided_transients`.

    Returns
    -------
    OrbitHistory

    Raises
    ------
    ValueError
        If there are no records or they belong to more than one object.
    """
    if not records:
        raise ValueError("no element sets given")
    ids = {r.norad_id for r in records}
    if len(ids) != 1:
        raise ValueError(f"element sets belong to several objects: {sorted(ids)}")

    merged = merge_refits(records, merge_within_s)
    regime = classify_regime(merged)
    epochs = [r.epoch_utc for r in merged]
    t = np.array([(e - epochs[0]).total_seconds() / 86400.0 for e in epochs])
    a = np.array([semi_major_axis_km(r) for r in merged])
    inc = np.array([r.inclination_rad for r in merged])
    ivec = np.array([inclination_vector_rad(r) for r in merged]).reshape(-1, 2)

    reject = np.zeros(len(merged), dtype=bool)
    if reject_transients:
        out_of_plane = ivec if regime == "geosynchronous" else inc
        reject = two_sided_transients(
            t, a, side_days, transient_sigmas
        ) | two_sided_transients(t, out_of_plane, side_days, transient_sigmas)

    keep = ~reject
    quality = DataQuality(
        records_in=len(records),
        after_merging_refits=len(merged),
        transients_rejected=int(reject.sum()),
        transient_epochs_utc=tuple(e for e, r in zip(epochs, reject, strict=True) if r),
        merge_within_s=merge_within_s,
    )
    kept_epochs = tuple(e for e, k in zip(epochs, keep, strict=True) if k)
    t_kept = t[keep] - t[keep][0] if keep.any() else t[keep]
    return OrbitHistory(
        norad_id=ids.pop(),
        epochs_utc=kept_epochs,
        t_days=t_kept,
        a_km=a[keep],
        inclination_rad=inc[keep],
        inclination_vector_rad=ivec[keep],
        regime=regime,
        quality=quality,
    )
