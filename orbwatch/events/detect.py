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

from orbwatch.events.history import MIN_GAP_DAYS, OrbitHistory, Regime, robust_sigma

DEFAULT_SIGMAS: float = 5.0
DEFAULT_RATE_WINDOW_DAYS: float = 5.0

DEFAULT_MIN_DELTA_V_M_S: dict[tuple[str, str], float] = {
    ("in_plane", "other"): 0.05,
    ("in_plane", "geosynchronous"): 0.05,
    ("out_of_plane", "other"): 0.30,
    ("out_of_plane", "geosynchronous"): 0.20,
}
"""Smallest event treated as a manoeuvre, m/s, by signal and orbit regime.

Element set residuals have heavy tails, so a five-sigma threshold alone still
passes fit noise. Each floor sits just above the largest spurious event seen on
a satellite that cannot have manoeuvred on that axis, over one year to
2026-09-21:

- In-plane: every ISS detection not matching an announced reboost was at most
  0.04 m/s, against 0.25 m/s for the smallest real one, and Hubble, which has no
  propulsion, showed no impulsive in-plane event at all.
- Out-of-plane, low orbit: Hubble showed spurious inclination steps up to
  0.26 m/s. Every out-of-plane event on the ISS and Tiangong was below that.
- Out-of-plane, geostationary: INTELSAT 905, no longer held in inclination,
  showed spurious steps up to 0.17 m/s, while every one of INMARSAT 5-F3's 18
  north-south events was at least 0.2 m/s.

Calibrated on those negative controls, not tuned on the answers; callers can
override them.
"""

FLOOR_OVERRIDE_SIGMAS: dict[str, float] = {"in_plane": 30.0, "out_of_plane": 100.0}
"""Significance at which an event is kept whatever its size, by signal.

The delta-v floors guard against heavy-tailed fit noise, which scales with
how noisy an object's element sets are. A fixed floor in m/s is therefore
too strict for clean histories: NOAA 20 scatters 0.1 m per gap in semi-major
axis, 40 times less than the ISS, and its 57 m raise of 28 January 2026, at
537 sigma, fell under the 0.05 m/s floor. INTELSAT 905 lost most of its
east-west burns the same way: 0.03 to 0.05 m/s each, all raising, roughly
weekly, at up to 236 sigma.

The overrides sit about twice above the largest spurious event on any control
over the year to 2026-09-22: in plane, 16 sigma (ISS, Hubble, Tiangong); out
of plane, 48 sigma, from INTELSAT 905, whose largest out-of-plane steps share
their windows with its east-west burns and are side effects of them, not
north-south manoeuvres.
"""

TLE_ANGLE_RESOLUTION_DEG: float = 1e-4
TLE_MEAN_MOTION_RESOLUTION_REV_PER_DAY: float = 1e-8
"""Last digits of the TLE inclination, node and mean motion fields.

A difference of two independently rounded values has a standard deviation of
the resolution over the square root of six. The per-gap scatter is never
estimated below that, because a slowly changing element can round to the same
value for most gaps, and a robust scatter of zero would then flag every
single rounding step. NOAA 20's inclination did exactly that.
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

EXCURSION_WINDOW_DAYS: float = 10.0
EXCURSION_CANCELLATION: float = 0.3
"""Two events form an excursion when the second undoes the first.

Specifically, when the second starts within ``EXCURSION_WINDOW_DAYS`` of the
first ending and the net of the two is at most this fraction of the smaller.
A run of bad element sets lasting several days can survive cleaning, and then
its entry and exit show up as two separate, individually coherent steps. On
INMARSAT 5-F3 one such run, around 1 January 2026, sat at zero inclination for
several days. A real burn exactly undone within ten days is possible but rare,
and is the documented blind spot of this rule.
"""

Kind = Literal["in_plane", "out_of_plane"]
Profile = Literal["impulsive", "sustained"]


@dataclass(frozen=True)
class StepAnalysis:
    """Per-gap working for one signal, kept so budgets and plots can reuse it.

    Arrays have one row per gap between consecutive samples. ``usable`` gaps
    are long enough to carry a rate. ``edge`` gaps sit at the start of the
    history with no past to estimate their natural rate from, and are judged
    against a raised per-gap threshold; ``gap_threshold`` equals ``threshold``
    everywhere else.
    """

    kind: Kind
    dt_days: NDArray[np.float64]
    usable: NDArray[np.bool_]
    edge: NDArray[np.bool_]
    local_rate: NDArray[np.float64]
    residual: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    sigma: float
    threshold: float
    gap_threshold: NDArray[np.float64]
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
    excursion : bool
        True when a later event undoes this one. See ``EXCURSION_CANCELLATION``.
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
    excursion: bool = False

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


def _resolution_sigma(history: OrbitHistory, kind: Kind) -> float:
    """Scatter that TLE rounding alone puts into one gap's change."""
    if kind == "out_of_plane":
        return float(np.deg2rad(TLE_ANGLE_RESOLUTION_DEG)) / np.sqrt(6.0)
    a_km = float(np.mean(history.a_km))
    n_rev_day = np.sqrt(history.mu_km3_s2 / a_km**3) * 86400.0 / (2.0 * np.pi)
    quantum_km = a_km * (2.0 / 3.0) * TLE_MEAN_MOTION_RESOLUTION_REV_PER_DAY / n_rev_day
    return float(quantum_km / np.sqrt(6.0))


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

    Edge gaps, at the very start of a history, have a look-back window that
    reaches before the first sample and holds fewer than three usable gaps.
    There is no past to estimate their natural rate from, so the global median
    stands in, and it stands in badly wherever the rate varies. Their threshold
    is raised by the largest departure of any local rate from the global one,
    times the gap length: the step must exceed what not knowing the rate could
    produce. On the ISS the first gap of the year to 2026-09-22 spanned a
    40-hour silence just before a reboost, when the low orbit was decaying
    about three times faster than the yearly median, and without this it was
    flagged as a 154 m lowering burn. A reboost at the edge still clears the
    raised threshold by a wide margin.
    """
    x = _signal(history, kind)
    t = history.t_days
    dt = np.diff(t)
    sigma_floor = _resolution_sigma(history, kind)
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
    no_past = t[:-1] - rate_window_days < t[0]

    flagged = np.zeros(dt.size, dtype=bool)
    for _ in range(2):
        local = np.empty_like(change)
        thin = np.zeros(dt.size, dtype=bool)
        for k in range(dt.size):
            window = np.arange(first[k], last[k])
            window = (
                window[usable[window] & ~flagged[window]] if window.size else window
            )
            if window.size >= 3:
                local[k] = np.median(rates[window], axis=0)
            else:
                local[k] = global_rate
                thin[k] = True
        edge = usable & thin & no_past
        judged = usable & ~edge if (usable & ~edge).any() else usable
        residual = change - local * dt[:, None]
        if x.shape[1] == 1:
            centre = np.median(residual[judged, 0])
            magnitude = np.abs(residual[:, 0] - centre)
            sigma = max(robust_sigma(residual[judged, 0]), sigma_floor)
            threshold = sigmas * sigma
        else:
            magnitude = np.linalg.norm(residual, axis=1)
            sigma = max(robust_sigma(magnitude[judged]), sigma_floor)
            threshold = float(np.median(magnitude[judged]) + sigmas * sigma)
        spread = float(np.linalg.norm(local[judged] - global_rate, axis=1).max())
        gap_threshold = threshold + np.where(edge, spread * dt, 0.0)
        flagged = usable & (magnitude > gap_threshold)

    return StepAnalysis(
        kind=kind,
        dt_days=dt,
        usable=usable,
        edge=edge,
        local_rate=local,
        residual=residual,
        magnitude=magnitude,
        sigma=float(sigma),
        threshold=float(threshold),
        gap_threshold=gap_threshold,
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
    return _mark_excursions(events)


def _mark_excursions(events: list[Manoeuvre]) -> list[Manoeuvre]:
    """Flag pairs of events where the second returns the orbit to where it was."""
    from dataclasses import replace

    flagged = [False] * len(events)
    for i, first in enumerate(events):
        if flagged[i] or first.coherence < MIN_COHERENCE:
            continue
        a = np.asarray(first.change)
        for j in range(i + 1, len(events)):
            second = events[j]
            gap_days = (second.start_utc - first.end_utc).total_seconds() / 86400.0
            if gap_days > EXCURSION_WINDOW_DAYS:
                break
            if flagged[j] or second.coherence < MIN_COHERENCE:
                continue
            b = np.asarray(second.change)
            smaller = min(np.linalg.norm(a), np.linalg.norm(b))
            if (
                smaller > 0
                and np.linalg.norm(a + b) <= EXCURSION_CANCELLATION * smaller
            ):
                flagged[i] = flagged[j] = True
                break
    return [replace(e, excursion=f) for e, f in zip(events, flagged, strict=True)]


def floor_m_s(kind: str, regime: str, override: float | None = None) -> float:
    """The delta-v floor for one signal and regime, or the override if given."""
    return DEFAULT_MIN_DELTA_V_M_S[(kind, regime)] if override is None else override


def clears_floor(
    event: Manoeuvre, regime: Regime, override: float | None = None
) -> bool:
    """At or above the delta-v floor, or significant enough to keep regardless.

    See ``DEFAULT_MIN_DELTA_V_M_S`` and ``FLOOR_OVERRIDE_SIGMAS``.
    """
    return (
        event.delta_v_m_s >= floor_m_s(event.kind, regime, override)
        or event.significance >= FLOOR_OVERRIDE_SIGMAS[event.kind]
    )


def is_drag_surge(event: Manoeuvre, regime: Regime) -> bool:
    """A sustained drop in semi-major axis outside GEO: drag, not a burn.

    The detector flags any change the recent decay rate does not explain. When
    a geomagnetic storm heats the upper atmosphere, drag rises for days and
    every low orbit drops faster than its recent rate at once. That is real
    and worth reporting, but it is natural. On 19 to 22 January 2026, during
    a G4 storm (NOAA SWPC), the ISS lost an extra 191 m, Tiangong a similar
    amount and Hubble, which has no propulsion, 141 m. A low-thrust lowering
    would look the same, and is rare enough to accept as the blind spot.
    """
    return (
        regime != "geosynchronous"
        and event.kind == "in_plane"
        and event.profile == "sustained"
        and event.change[0] < 0.0
    )


def _significant_events(
    history: OrbitHistory,
    sigmas: float,
    rate_window_days: float,
    min_delta_v_m_s: float | None,
) -> list[Manoeuvre]:
    """Coherent events at or above the floor that are not returned excursions."""
    found: list[Manoeuvre] = []
    for kind in ("in_plane", "out_of_plane"):
        analysis = analyse_steps(history, kind, sigmas, rate_window_days)
        found.extend(
            m
            for m in manoeuvres_from(history, analysis)
            if clears_floor(m, history.regime, min_delta_v_m_s)
            and m.coherence >= MIN_COHERENCE
            and not m.excursion
        )
    return sorted(found, key=lambda m: m.start_utc)


def detect_manoeuvres(
    history: OrbitHistory,
    sigmas: float = DEFAULT_SIGMAS,
    rate_window_days: float = DEFAULT_RATE_WINDOW_DAYS,
    min_delta_v_m_s: float | None = None,
) -> list[Manoeuvre]:
    """All in-plane and out-of-plane manoeuvres at or above the floor, in time order.

    ``min_delta_v_m_s`` overrides the calibrated per-regime floors in
    ``DEFAULT_MIN_DELTA_V_M_S`` when given. Drag surges are not manoeuvres; see
    :func:`detect_drag_surges`.
    """
    return [
        m
        for m in _significant_events(history, sigmas, rate_window_days, min_delta_v_m_s)
        if not is_drag_surge(m, history.regime)
    ]


def detect_drag_surges(
    history: OrbitHistory,
    sigmas: float = DEFAULT_SIGMAS,
    rate_window_days: float = DEFAULT_RATE_WINDOW_DAYS,
    min_delta_v_m_s: float | None = None,
) -> list[Manoeuvre]:
    """Sustained drops outside GEO that the recent decay does not explain.

    See :func:`is_drag_surge`. Reported so they are visible, and counted as
    natural loss in the in-plane budget, never as propellant.
    """
    return [
        m
        for m in _significant_events(history, sigmas, rate_window_days, min_delta_v_m_s)
        if is_drag_surge(m, history.regime)
    ]
