"""One call from element sets to manoeuvres, data quality and budgets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from orbwatch.catalog.tle import TLE
from orbwatch.events.budget import Budget, in_plane_budget, out_of_plane_budget
from orbwatch.events.detect import (
    DEFAULT_MIN_DELTA_V_M_S,
    DEFAULT_RATE_WINDOW_DAYS,
    DEFAULT_SIGMAS,
    MIN_COHERENCE,
    Manoeuvre,
    StepAnalysis,
    analyse_steps,
    manoeuvres_from,
)
from orbwatch.events.history import OrbitHistory, build_history


@dataclass(frozen=True)
class EventReport:
    """Everything the events layer knows about one object's history."""

    history: OrbitHistory
    in_plane: StepAnalysis
    out_of_plane: StepAnalysis
    manoeuvres: tuple[Manoeuvre, ...]
    below_floor: tuple[Manoeuvre, ...]
    incoherent: tuple[Manoeuvre, ...]
    in_plane_budget: Budget
    out_of_plane_budget: Budget


def analyse(
    records: Sequence[TLE],
    sigmas: float = DEFAULT_SIGMAS,
    rate_window_days: float = DEFAULT_RATE_WINDOW_DAYS,
    min_delta_v_m_s: float = DEFAULT_MIN_DELTA_V_M_S,
    **cleaning: float,
) -> EventReport:
    """Clean a history, detect manoeuvres and close the budgets.

    Parameters
    ----------
    records : sequence of TLE
        Element set history for one object.
    sigmas : float, optional
        Detection threshold in robust standard deviations.
    rate_window_days : float, optional
        Look-back for the local natural rate.
    min_delta_v_m_s : float, optional
        Detections below this are reported separately as ``below_floor`` and
        left out of the budgets.

    Detections whose per-gap steps cancel are reported as ``incoherent``: they
    are transients the cleaning step did not catch, never manoeuvres.
    **cleaning
        Passed to :func:`orbwatch.events.history.build_history`.
    """
    history = build_history(records, **cleaning)
    in_plane = analyse_steps(history, "in_plane", sigmas, rate_window_days)
    out_of_plane = analyse_steps(history, "out_of_plane", sigmas, rate_window_days)
    candidates = manoeuvres_from(history, in_plane) + manoeuvres_from(
        history, out_of_plane
    )
    candidates.sort(key=lambda m: m.start_utc)
    incoherent = [m for m in candidates if m.coherence < MIN_COHERENCE]
    coherent = [m for m in candidates if m.coherence >= MIN_COHERENCE]
    events = [m for m in coherent if m.delta_v_m_s >= min_delta_v_m_s]
    return EventReport(
        history=history,
        in_plane=in_plane,
        out_of_plane=out_of_plane,
        manoeuvres=tuple(events),
        below_floor=tuple(m for m in coherent if m.delta_v_m_s < min_delta_v_m_s),
        incoherent=tuple(incoherent),
        in_plane_budget=in_plane_budget(history, in_plane, events),
        out_of_plane_budget=out_of_plane_budget(history, out_of_plane, events),
    )
