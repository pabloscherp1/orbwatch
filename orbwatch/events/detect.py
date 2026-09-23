"""Manoeuvre detection: steps in an orbit history that nature cannot explain.

For each gap between consecutive samples, the expected natural change is the
local rate of the recent past times the gap length. Whatever is left over is the
residual. A residual larger than a few robust standard deviations of all
residuals is a candidate burn; adjacent flagged gaps merge into one manoeuvre.

Two signals, analysed independently:

- **In-plane**, from the semi-major axis: reboosts, orbit raising, east-west
  station-keeping. The natural change is drag decay in low orbit.
- **Out-of-plane**, from the inclination vector at GEO or the inclination
  elsewhere: north-south station-keeping and plane changes.

Validated on the ISS against NASA's announced reboosts for the year to
2026-09-21: 10 of 10 found, published altitude raises matched within 2 to 4%.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orbwatch.events.history import MIN_GAP_DAYS, OrbitHistory, robust_sigma

DEFAULT_SIGMAS: float = 5.0
DEFAULT_RATE_WINDOW_DAYS: float = 5.0

DEFAULT_MIN_DELTA_V_M_S: float = 0.05
"""Events smaller than this are treated as fit noise rather than manoeuvres.

Element set residuals have heavy tails, so a five-sigma threshold alone still
passes some noise. On a year of ISS history every detection not matching an
announced reboost was 0.04 m/s or less, while the smallest real reboost was
0.25 m/s. The floor sits between the two. It is a choice informed by that one
validation set, and callers can change it.
"""

MIN_COHERENCE: float = 0.5
"""Below this, an event's per-gap steps mostly cancel: a transient, not a burn.

Coherence is the size of the net change divided by the sum of the per-gap sizes.
A burn spread over two element sets points the same way twice, close to 1. A bad
element set that sits off the path produces a step away and a step back, close
to 0. On the ISS, a fit 45 m low produced two opposing 5 sigma steps with a
coherence of 0.06.
"""

IMPULSIVE_FRACTION: float = 0.6
"""An event is impulsive when one gap carries at least this share of its change.

Anything else is sustained: a change spread across several gaps in proportion to
their length. In low orbit that is usually a drag surge from space weather, and
at GEO it may be a low-thrust manoeuvre, so it is labelled, not discarded.
"""

Kind = Literal["in_plane", "out_of_plane"]
Profile = Literal["impulsive", "sustained"]


@dataclass(frozen=True)
class StepAnalysis:
    """Per-gap working for one signal, kept so budgets and plots can reuse it.

    Arrays have one row per gap between consecutive samples.
    """

    kind: Kind
    dt_days: NDArray[np.float64]
    usable: NDArray[np.bool_]
    local_rate: NDArray[np.float64]
    residual: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    sigma: float
    threshold: float
    flagged: NDArray[np.bool_]


@dataclass(frozen=True)
class Manoeuvre:
    """One detected manoeuvre.

    Attributes
    ----------
    kind : {"in_plane", "out_of_plane"}
    start_utc, end_utc : datetime
        Epochs of the last sample before and the first sample after. The burn
        happened somewhere in between, and not necessarily near the middle: an
        element set's epoch is not the time its information comes from.
    change : tuple of float
        Net change across the manoeuvre, after removing natural drift. Semi-major
        axis in km for in-plane; inclination vector or inclination in radians for
        out-of-plane.
    delta_v_m_s : float
        Speed change implied, m/s, summed gap by gap so opposing components
        never cancel.
    significance : float
        Largest per-gap residual in robust standard deviations.
    gaps : int
        Number of adjacent gaps merged.
    coherence : float
        Net change over summed per-gap change, 0 to 1. See ``MIN_COHERENCE``.
    profile : {"impulsive", "sustained"}
        See ``IMPULSIVE_FRACTION``.
    """

    kind: Kind
    start_utc: datetime
    end_utc: datetime
    change: tuple[float, ...]
    delta_v_m_s: float
    significance: float
    gaps: int
    coherence: float
    profile: Profile

    @property
    def window_hours(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds() / 3600.0

    @property
    def delta_a_km(self) -> float | None:
        return self.change[0] if self.kind == "in_plane" else None

    @property
    def delta_i_rad(self) -> float | None:
        return (
            float(np.linalg.norm(self.change)) if self.kind == "out_of_plane" else None
        )


def _signal(history: OrbitHistory, kind: Kind) -> NDArray[np.float64]:
    if kind == "in_plane":
        return history.a_km[:, None]
    return history.out_of_plane_signal


def analyse_steps(
    history: OrbitHistory,
    kind: Kind,
    sigmas: float = DEFAULT_SIGMAS,
    rate_window_days: float = DEFAULT_RATE_WINDOW_DAYS,
) -> StepAnalysis:
    """Residual change per gap and which gaps are flagged.

    Parameters
    ----------
    history : OrbitHistory
    kind : {"in_plane", "out_of_plane"}
    sigmas : float, optional
        Detection threshold in robust standard deviations.
    rate_window_days : float, optional
        How far back the local natural rate is estimated from.

    Notes
    -----
    The local rate for a gap is the median rate over usable, unflagged gaps that
    ended within ``rate_window_days`` before it starts. It is computed twice: the
    second pass excludes gaps flagged by the first, so a burn does not
    contaminate the drift estimate of the gaps after it. Using a local rate
    rather than one global value matters in low orbit, where drag changes with
    solar activity and with altitude after every reboost.

    Gaps shorter than an hour are never flagged and never feed a rate, because
    a small step divided by a tiny interval manufactures a huge apparent rate.
    """
    x = _signal(history, kind)
    t = history.t_days
    dt = np.diff(t)
    change = np.diff(x, axis=0)
    usable = dt > MIN_GAP_DAYS
    rates = np.where(
        usable[:, None], change / np.where(usable, dt, 1.0)[:, None], np.nan
    )
    global_rate = (
        np.nanmedian(rates[usable], axis=0) if usable.any() else np.zeros(x.shape[1])
    )

    gap_end = t[1:]
    first = np.searchsorted(gap_end, t[:-1] - rate_window_days, side="right")
    last = np.searchsorted(gap_end, t[:-1], side="right")

    flagged = np.zeros(dt.size, dtype=bool)
    for _ in range(2):
        local = np.empty_like(change)
        for k in range(dt.size):
            window = np.arange(first[k], last[k])
            window = (
                window[usable[window] & ~flagged[window]] if window.size else window
            )
            local[k] = (
                np.median(rates[window], axis=0) if window.size >= 3 else global_rate
            )
        residual = change - local * dt[:, None]
        if x.shape[1] == 1:
            centre = np.median(residual[usable, 0])
            magnitude = np.abs(residual[:, 0] - centre)
            sigma = robust_sigma(residual[usable, 0])
            threshold = sigmas * sigma
        else:
            magnitude = np.linalg.norm(residual, axis=1)
            sigma = robust_sigma(magnitude[usable])
            threshold = float(np.median(magnitude[usable]) + sigmas * sigma)
        flagged = usable & (magnitude > threshold)

    return StepAnalysis(
        kind=kind,
        dt_days=dt,
        usable=usable,
        local_rate=local,
        residual=residual,
        magnitude=magnitude,
        sigma=float(sigma),
        threshold=float(threshold),
        flagged=flagged,
    )


def _delta_v_m_s(
    history: OrbitHistory, kind: Kind, residual: NDArray[np.float64]
) -> float:
    """Speed change for a set of per-gap residuals, summed by magnitude.

    In-plane: a tangential impulse changes the semi-major axis by
    da = 2 a dv / v for a near-circular orbit, so dv = (v/2) |da| / a.
    Out-of-plane: a small plane change costs dv = v |di|.
    """
    v_m_s = history.mean_speed_km_s * 1000.0
    if kind == "in_plane":
        return float(0.5 * v_m_s * np.abs(residual[:, 0]).sum() / np.mean(history.a_km))
    return float(v_m_s * np.linalg.norm(residual, axis=1).sum())


def manoeuvres_from(history: OrbitHistory, analysis: StepAnalysis) -> list[Manoeuvre]:
    """Merge adjacent flagged gaps into manoeuvres."""
    groups: list[list[int]] = []
    for k in np.flatnonzero(analysis.flagged):
        if groups and k == groups[-1][-1] + 1:
            groups[-1].append(int(k))
        else:
            groups.append([int(k)])

    events = []
    for g in groups:
        residual = analysis.residual[g]
        significance = (
            analysis.magnitude[g].max() / analysis.sigma
            if analysis.sigma > 0
            else float("inf")
        )
        per_gap = np.linalg.norm(residual, axis=1)
        net = float(np.linalg.norm(residual.sum(axis=0)))
        coherence = net / float(per_gap.sum()) if per_gap.sum() > 0 else 0.0
        largest_share = float(per_gap.max()) / net if net > 0 else 0.0
        profile: Profile = (
            "impulsive" if largest_share >= IMPULSIVE_FRACTION else "sustained"
        )
        events.append(
            Manoeuvre(
                kind=analysis.kind,
                start_utc=history.epochs_utc[g[0]],
                end_utc=history.epochs_utc[g[-1] + 1],
                change=tuple(float(v) for v in residual.sum(axis=0)),
                delta_v_m_s=_delta_v_m_s(history, analysis.kind, residual),
                significance=float(significance),
                gaps=len(g),
                coherence=coherence,
                profile=profile,
            )
        )
    return events


def detect_manoeuvres(
    history: OrbitHistory,
    sigmas: float = DEFAULT_SIGMAS,
    rate_window_days: float = DEFAULT_RATE_WINDOW_DAYS,
    min_delta_v_m_s: float = DEFAULT_MIN_DELTA_V_M_S,
) -> list[Manoeuvre]:
    """All in-plane and out-of-plane manoeuvres at or above the floor, in time order."""
    found: list[Manoeuvre] = []
    for kind in ("in_plane", "out_of_plane"):
        analysis = analyse_steps(history, kind, sigmas, rate_window_days)
        found.extend(
            m
            for m in manoeuvres_from(history, analysis)
            if m.delta_v_m_s >= min_delta_v_m_s and m.coherence >= MIN_COHERENCE
        )
    return sorted(found, key=lambda m: m.start_utc)
