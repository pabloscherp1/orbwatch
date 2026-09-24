"""Pattern of life: an operator's east-west station-keeping, estimated from its
own history, used to predict the next burn, and scored against what happened.

A geostationary satellite away from the two stable longitudes is pulled east or
west by the ellipticity of the equator. Held in a longitude box, it traces a
parabola: a burn sends it drifting away from one edge, the pull slows it, turns
it round and brings it back, and when it reaches that edge again the operator
burns again. Physics sets the curvature of the parabola and the operator's
policy sets the box and where the burn is triggered, so both can be estimated,
and the next burn predicted from where the current arc meets the trigger.

Two predictors are compared, each using only what was known at the time:

- **interval**: the last burn plus the recent median interval between burns.
- **trigger**: fit the current arc with the estimated curvature and solve for
  when it reaches the recent trigger longitude.

Whichever has the better walk-forward record on an object is the one used to
forecast it. On INTELSAT 905 over the year to 2026-09-22 that was the interval,
with a median error of 1.4 days: the operator's trigger longitude wanders by
about 0.010 deg, which at the ~0.007 deg/day drift near the box edge is itself
1.4 days, so no predictor built on the trigger can do better. The physics
explains the cycle; the operator's variability limits the forecast.

Units follow the names: degrees and days, except where stated.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from orbwatch.events.detect import Manoeuvre
from orbwatch.events.history import OrbitHistory, robust_sigma

J22: float = 1.8155e-6
"""Second-degree, second-order sectoral coefficient of Earth's gravity field."""

J22_LONGITUDE_DEG: float = -14.93
"""Longitude of the equator's long axis. The stable points sit 90 deg either
side of it, near 75 E and 105 W, and the unstable points on it and opposite."""

J22_AMPLITUDE_DEG_PER_DAY2: float = float(
    np.rad2deg(18.0 * J22 * (6378.137 / 42164.17) ** 2 * 7.2921158553e-5**2)
    * 86400.0**2
)
"""Largest longitude acceleration from J22, 18 J22 (R/a)^2 w^2: 1.70e-3 deg/day^2."""

MIN_BURNS: int = 6
RECENT_BURNS: int = 5
MIN_ARC_SAMPLES: int = 6
MIN_ARC_DAYS: float = 2.0

Method = Literal["interval", "trigger"]


def triaxiality_acceleration_deg_per_day2(longitude_deg: float) -> float:
    """Longitude acceleration of a geostationary satellite from J22 alone.

    Positive is eastward. Zero at the stable points near 75.1 E and 104.9 W,
    which it pushes towards, and at the unstable points near 14.9 W and
    165.1 E. The higher harmonics J31 and J33 shift these by a fraction of a
    degree and change the size by some ten percent, and the Sun and Moon add
    periodic terms, so this is a reference, not a prediction.
    """
    return J22_AMPLITUDE_DEG_PER_DAY2 * float(
        np.sin(2.0 * np.deg2rad(longitude_deg - J22_LONGITUDE_DEG))
    )


def longitude_deg(history: OrbitHistory) -> NDArray[np.float64]:
    """Mean geographic longitude, degrees, continuous across the date line."""
    if history.mean_longitude_rad is None:
        raise ValueError("mean longitude is only defined for geosynchronous objects")
    return np.rad2deg(np.unwrap(history.mean_longitude_rad))


def east_west_burns(manoeuvres: Sequence[Manoeuvre]) -> list[Manoeuvre]:
    """Impulsive in-plane manoeuvres, in time order."""
    return sorted(
        (m for m in manoeuvres if m.kind == "in_plane" and m.profile == "impulsive"),
        key=lambda m: m.start_utc,
    )


def _days(history: OrbitHistory, when: datetime) -> float:
    return (when - history.epochs_utc[0]).total_seconds() / 86400.0


def _last_index_at_or_before(history: OrbitHistory, when: datetime) -> int:
    return bisect_right(history.epochs_utc, when) - 1


@dataclass(frozen=True)
class EastWestPolicy:
    """An operator's east-west station-keeping, estimated from its history.

    Attributes
    ----------
    burns : int
        Burns the estimate is based on.
    interval_days, interval_spread_days : float
        Median and robust spread of the recent intervals between burns.
    delta_v_m_s, delta_a_km : float
        Median recent burn.
    trigger_longitude_deg, trigger_spread_deg : float
        Median and robust spread of the longitude at the last element set
        before each recent burn: where the operator burns.
    acceleration_deg_per_day2 : float
        Median curvature of the arcs between burns, twice the quadratic
        coefficient of a fit to each arc.
    acceleration_arcs : int
        Arcs long enough to fit.
    model_acceleration_deg_per_day2 : float
        J22 alone at the mean longitude, for comparison.
    box_deg : tuple of float
        2nd and 98th percentiles of the longitude since the oldest recent burn.
    """

    burns: int
    interval_days: float
    interval_spread_days: float
    delta_v_m_s: float
    delta_a_km: float
    trigger_longitude_deg: float
    trigger_spread_deg: float
    acceleration_deg_per_day2: float
    acceleration_arcs: int
    model_acceleration_deg_per_day2: float
    box_deg: tuple[float, float]

    @property
    def edge_drift_deg_per_day(self) -> float:
        """Drift rate on reaching the box edge from rest at the far edge."""
        width = self.box_deg[1] - self.box_deg[0]
        return float(np.sqrt(2.0 * abs(self.acceleration_deg_per_day2) * width))

    @property
    def timing_floor_days(self) -> float:
        """Timing scatter implied by the scatter of the trigger longitude alone.

        No predictor built on the trigger can beat it, however good its physics.
        """
        drift = self.edge_drift_deg_per_day
        return self.trigger_spread_deg / drift if drift > 0 else float("inf")


def estimate_east_west_policy(
    history: OrbitHistory,
    manoeuvres: Sequence[Manoeuvre],
    until_utc: datetime | None = None,
    recent: int = RECENT_BURNS,
) -> EastWestPolicy | None:
    """Estimate the policy from burns and element sets up to ``until_utc``.

    Returns None outside GEO, with fewer than ``MIN_BURNS`` burns, or when no
    arc between burns is long enough to measure its curvature.
    """
    if history.mean_longitude_rad is None:
        return None
    burns = [
        m
        for m in east_west_burns(manoeuvres)
        if until_utc is None or m.end_utc <= until_utc
    ]
    if len(burns) < MIN_BURNS:
        return None
    lon = longitude_deg(history)
    t = history.t_days
    last = (
        len(t)
        if until_utc is None
        else _last_index_at_or_before(history, until_utc) + 1
    )

    curvatures = []
    for before, after in zip(burns, burns[1:], strict=False):
        arc = (t >= _days(history, before.end_utc)) & (
            t <= _days(history, after.start_utc)
        )
        arc[last:] = False
        if arc.sum() >= MIN_ARC_SAMPLES and np.ptp(t[arc]) >= MIN_ARC_DAYS:
            tau = t[arc] - t[arc][0]
            curvatures.append(2.0 * np.polyfit(tau, lon[arc], 2)[0])
    if not curvatures:
        return None

    recent_burns = burns[-recent:]
    starts = np.array([_days(history, b.start_utc) for b in burns])
    intervals = np.diff(starts)[-recent:]
    triggers = np.array(
        [lon[_last_index_at_or_before(history, b.start_utc)] for b in recent_burns]
    )
    since = t >= _days(history, recent_burns[0].start_utc)
    since[last:] = False
    box = np.percentile(lon[since], [2.0, 98.0])
    return EastWestPolicy(
        burns=len(burns),
        interval_days=float(np.median(intervals)),
        interval_spread_days=float(robust_sigma(intervals)),
        delta_v_m_s=float(np.median([b.delta_v_m_s for b in recent_burns])),
        delta_a_km=float(np.median([b.change[0] for b in recent_burns])),
        trigger_longitude_deg=float(np.median(triggers)),
        trigger_spread_deg=float(robust_sigma(triggers)),
        acceleration_deg_per_day2=float(np.median(curvatures)),
        acceleration_arcs=len(curvatures),
        model_acceleration_deg_per_day2=triaxiality_acceleration_deg_per_day2(
            float(np.mean(lon[:last]))
        ),
        box_deg=(float(box[0]), float(box[1])),
    )


@dataclass(frozen=True)
class BurnPrediction:
    """A predicted next burn, and the arc it was predicted from.

    ``arc`` holds (start epoch, longitude at start, drift rate at start,
    curvature) of the fitted current arc, so the fit can be drawn; None for the
    interval method or when the trigger method fell back to it.
    """

    method: Method
    made_at_utc: datetime
    predicted_utc: datetime
    fell_back: bool = False
    arc: tuple[datetime, float, float, float] | None = None


def predict_next_burn(
    history: OrbitHistory,
    policy: EastWestPolicy,
    last_burn: Manoeuvre,
    at_utc: datetime,
    method: Method,
) -> BurnPrediction:
    """Predict the burn after ``last_burn`` using data up to ``at_utc`` only.

    The trigger method fits the element sets since ``last_burn`` with the
    policy's curvature fixed, which leaves two unknowns and needs two samples,
    then takes the later time the arc reaches the trigger longitude. With too
    few samples, or an arc that never reaches the trigger, it falls back to
    the interval method and says so.
    """
    by_interval = last_burn.start_utc + timedelta(days=policy.interval_days)
    if method == "interval":
        return BurnPrediction("interval", at_utc, by_interval)

    lon = longitude_deg(history)
    t = history.t_days
    arc = (t >= _days(history, last_burn.end_utc)) & (t <= _days(history, at_utc))
    if arc.sum() < 2:
        return BurnPrediction("trigger", at_utc, by_interval, fell_back=True)
    t0 = t[arc][0]
    tau = t[arc] - t0
    accel = policy.acceleration_deg_per_day2
    rate, lon0 = np.polyfit(tau, lon[arc] - 0.5 * accel * tau**2, 1)

    roots = np.roots([0.5 * accel, rate, lon0 - policy.trigger_longitude_deg])
    real = roots[np.isreal(roots)].real
    if accel == 0.0 or real.size == 0:
        return BurnPrediction("trigger", at_utc, by_interval, fell_back=True)
    now = _days(history, at_utc) - t0
    crossing = max(float(real.max()), now)
    start = history.epochs_utc[0] + timedelta(days=float(t0))
    return BurnPrediction(
        "trigger",
        at_utc,
        start + timedelta(days=crossing),
        arc=(start, float(lon0), float(rate), float(accel)),
    )


@dataclass(frozen=True)
class BacktestEntry:
    """One held-out burn, and what each method predicted for it.

    The truth is the middle of the burn's window, between the last element set
    before it and the first after, and ``truth_half_width_days`` is how far the
    real burn can be from that middle: errors below it are not meaningful.
    Errors are predicted minus true, days: negative is early.
    """

    truth_utc: datetime
    truth_half_width_days: float
    made_at_utc: datetime
    interval_error_days: float
    trigger_error_days: float
    trigger_fell_back: bool

    def error_days(self, method: Method) -> float:
        return (
            self.interval_error_days
            if method == "interval"
            else self.trigger_error_days
        )


@dataclass(frozen=True)
class Score:
    """How one method did over a backtest."""

    median_abs_days: float
    mean_abs_days: float
    within_one_day: float
    bias_days: float


@dataclass(frozen=True)
class Backtest:
    """Walk-forward scores over the held-out part of a history."""

    entries: tuple[BacktestEntry, ...]
    lead_days: float
    skipped: int
    """Held-out burns that came before a prediction could be made."""

    def score(self, method: Method) -> Score:
        errors = np.array([e.error_days(method) for e in self.entries])
        return Score(
            median_abs_days=float(np.median(np.abs(errors))),
            mean_abs_days=float(np.mean(np.abs(errors))),
            within_one_day=float(np.mean(np.abs(errors) <= 1.0)),
            bias_days=float(np.mean(errors)),
        )

    @property
    def truth_half_width_days(self) -> float:
        return float(np.median([e.truth_half_width_days for e in self.entries]))


def backtest_east_west(
    history: OrbitHistory,
    manoeuvres: Sequence[Manoeuvre],
    test_fraction: float = 1.0 / 3.0,
    lead_days: float = 1.0,
) -> Backtest | None:
    """Predict each burn in the last ``test_fraction`` of the history.

    Each prediction is made ``lead_days`` after the previous burn, from a
    policy estimated on burns and element sets up to that moment only. One
    caveat: the burns themselves come from a detection run over the whole
    history, whose noise estimate and transient test see a little of the
    future. The predictors do not.
    """
    burns = east_west_burns(manoeuvres)
    if len(burns) <= MIN_BURNS or len(history) < 2:
        return None
    cutoff = history.epochs_utc[0] + timedelta(
        days=(1.0 - test_fraction) * history.span_days
    )
    entries = []
    skipped = 0
    for k in range(MIN_BURNS, len(burns)):
        burn, previous = burns[k], burns[k - 1]
        if burn.start_utc < cutoff:
            continue
        made_at = previous.end_utc + timedelta(days=lead_days)
        if made_at >= burn.start_utc:
            skipped += 1
            continue
        policy = estimate_east_west_policy(history, burns[:k], until_utc=made_at)
        if policy is None:
            skipped += 1
            continue
        truth = burn.start_utc + (burn.end_utc - burn.start_utc) / 2
        errors = {}
        fell_back = False
        for method in ("interval", "trigger"):
            p = predict_next_burn(history, policy, previous, made_at, method)
            errors[method] = (p.predicted_utc - truth).total_seconds() / 86400.0
            fell_back = fell_back or p.fell_back
        entries.append(
            BacktestEntry(
                truth_utc=truth,
                truth_half_width_days=burn.window_hours / 48.0,
                made_at_utc=made_at,
                interval_error_days=errors["interval"],
                trigger_error_days=errors["trigger"],
                trigger_fell_back=fell_back,
            )
        )
    if not entries:
        return None
    return Backtest(entries=tuple(entries), lead_days=lead_days, skipped=skipped)


WELL_TIMED_HOURS: float = 24.0
"""Burn windows no longer than this are placed on a weekday."""


def weekday_counts(burns: Sequence[Manoeuvre]) -> tuple[int, ...]:
    """Burns per weekday, Monday first, counting windows under a day only.

    A window is the time between the last element set before a burn and the
    first after it; a longer one cannot say which day the burn was on. Operators
    often plan manoeuvres on a working-week schedule, and INTELSAT 905 burned on
    Mondays through April and May 2026.
    """
    counts = [0] * 7
    for b in burns:
        if b.window_hours <= WELL_TIMED_HOURS:
            counts[(b.start_utc + (b.end_utc - b.start_utc) / 2).weekday()] += 1
    return tuple(counts)


DRIFT_RELATIVE_TOLERANCE: float = 0.5
DRIFT_ABSOLUTE_TOLERANCE_DEG_PER_DAY2: float = 3e-4
"""How far the fitted curvature may sit from J22 alone before the arcs between
burns are judged not to be free drift.

The model assumes a satellite coasts between burns, so its longitude bends
under gravity alone. INTELSAT 905, kept by discrete chemical burns, fits 0.0011
against 0.0015 deg/day^2 from J22. INMARSAT 5-F3, steered by daily ion thruster
firings, fits 0.00006 against 0.0008: it never coasts, and a pattern built on
the few lumps in its control forecast with a median miss of 21 days on a
31-day cadence. The absolute part keeps the test meaningful near the stable
longitudes, where J22 alone predicts almost no pull. J31, J33, the Sun and the
Moon, left out of the model, stay inside the relative part.
"""

MIN_SKILL: float = 0.5
"""A forecast is shown only when its median miss on held-out burns is less than
this share of the interval between burns."""


def drifts_freely(policy: EastWestPolicy) -> bool:
    """Whether the arcs between burns bend the way gravity alone would."""
    model = policy.model_acceleration_deg_per_day2
    tolerance = max(
        DRIFT_RELATIVE_TOLERANCE * abs(model), DRIFT_ABSOLUTE_TOLERANCE_DEG_PER_DAY2
    )
    return abs(policy.acceleration_deg_per_day2 - model) <= tolerance


@dataclass(frozen=True)
class PatternOfLife:
    """An east-west pattern of life for one GEO object, or why there is none.

    ``applies`` is True only when the object burns discretely, drifts freely
    between burns, and the forecast has shown skill on held-out burns. When it
    is False, ``reason`` says which of those failed, in a sentence, and the
    fields that could be computed are still filled in so the numbers behind
    the verdict can be shown.

    ``forecast`` uses ``preferred``, the method with the lower median error in
    the shortest-lead backtest. ``trigger_forecast`` is kept as well, because
    its fitted arc is what shows the physics of the cycle.
    """

    applies: bool
    reason: str
    burns: int
    weekday_counts: tuple[int, ...]
    policy: EastWestPolicy | None = None
    preferred: Method = "interval"
    forecast: BurnPrediction | None = None
    trigger_forecast: BurnPrediction | None = None
    backtests: tuple[Backtest, ...] = ()


def pattern_of_life(
    history: OrbitHistory,
    manoeuvres: Sequence[Manoeuvre],
    lead_days: Sequence[float] = (1.0, 3.0),
) -> PatternOfLife | None:
    """The east-west pattern of life of a GEO object; None outside GEO.

    Always returns a verdict for a GEO object, so a missing pattern is
    explained rather than silently absent.
    """
    if history.mean_longitude_rad is None:
        return None
    burns = east_west_burns(manoeuvres)
    weekdays = weekday_counts(burns)
    if len(burns) < MIN_BURNS:
        return PatternOfLife(
            applies=False,
            reason=(
                f"Only {len(burns)} discrete east-west burn"
                f"{'' if len(burns) == 1 else 's'} in this span; a pattern needs"
                f" {MIN_BURNS}. The east-west keeping is either too small or too"
                " frequent to resolve burn by burn, or there is none."
            ),
            burns=len(burns),
            weekday_counts=weekdays,
        )
    policy = estimate_east_west_policy(history, manoeuvres)
    if policy is None:
        return PatternOfLife(
            applies=False,
            reason="No stretch between burns is long enough to measure the drift.",
            burns=len(burns),
            weekday_counts=weekdays,
        )

    now = history.epochs_utc[-1]
    backtests = tuple(
        b
        for b in (
            backtest_east_west(history, manoeuvres, lead_days=d) for d in lead_days
        )
        if b is not None
    )
    preferred: Method = "interval"
    if backtests:
        first = backtests[0]
        if (
            first.score("trigger").median_abs_days
            < first.score("interval").median_abs_days
        ):
            preferred = "trigger"

    reason = ""
    if not drifts_freely(policy):
        fitted = policy.acceleration_deg_per_day2
        model = policy.model_acceleration_deg_per_day2
        reason = (
            f"Between the detected burns the longitude does not drift freely: it"
            f" bends at {fitted:.5f} deg/day^2 where Earth's gravity alone gives"
            f" {model:.5f}."
            " The satellite is being steered continuously, so its east-west keeping"
            " is not a sequence of discrete burns and cannot be forecast as one."
        )
    elif backtests:
        miss = backtests[0].score(preferred).median_abs_days
        if miss >= MIN_SKILL * policy.interval_days:
            reason = (
                f"The best forecast missed held-out burns by a median {miss:.1f}"
                f" days against a {policy.interval_days:.1f}-day cadence: no useful"
                " skill."
            )
    return PatternOfLife(
        applies=not reason,
        reason=reason,
        burns=len(burns),
        weekday_counts=weekdays,
        policy=policy,
        preferred=preferred,
        forecast=predict_next_burn(history, policy, burns[-1], now, preferred),
        trigger_forecast=predict_next_burn(history, policy, burns[-1], now, "trigger"),
        backtests=backtests,
    )
