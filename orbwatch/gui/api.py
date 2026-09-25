"""JSON payloads for the ORBWATCH tracker interface.

Every number the interface displays is computed here, by the same tested
modules the rest of ORBWATCH uses. The browser only interpolates between
samples and draws. That split is deliberate: the alternative, running SGP4 in
JavaScript, would put the physics outside the test suite.

These functions are pure. They take parsed objects and return plain,
JSON-serialisable dictionaries, so they are tested without a server.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from orbwatch.access.lighting import (
    CIVIL_TWILIGHT_SUN_ELEVATION_RAD,
    is_sunlit,
    subsolar_point_rad,
    sun_position_km,
)
from orbwatch.access.passes import GroundSite, find_passes, horizon_half_angle_rad
from orbwatch.budget.budget import MARGINS, REQUIREMENTS, build_budget
from orbwatch.budget.budget import REFERENCE as BUDGET_REFERENCE
from orbwatch.catalog.frames import (
    ecef_to_geodetic,
    look_angles,
    teme_to_ecef,
    teme_to_ecef_position,
)
from orbwatch.catalog.propagate import StaleElementSetWarning, propagate
from orbwatch.catalog.timescales import SECONDS_PER_DAY, require_utc, time_grid
from orbwatch.catalog.tle import TLE
from orbwatch.events.analysis import EventReport, analyse
from orbwatch.events.budget import (
    ENDPOINT_WINDOW_DAYS,
    Budget,
    natural_inclination_path_rad,
)
from orbwatch.events.detect import (
    DEFAULT_SIGMAS,
    FLOOR_OVERRIDE_SIGMAS,
    Manoeuvre,
    StepAnalysis,
    floor_m_s,
)
from orbwatch.events.history import OrbitHistory
from orbwatch.events.pattern import (
    Score,
    east_west_burns,
    longitude_deg,
    pattern_of_life,
)
from orbwatch.rpo import (
    PROXIMITY_PERCENTILE,
    Dispersions,
    ProximityPlan,
    cw,
    plan_proximity,
)
from orbwatch.rpo.approach import ELLIPSE_RADIAL_M, INSPECTION_ORBITS, KEEP_OUT_M
from orbwatch.transfer.j2 import EARTH_RADIUS_KM
from orbwatch.transfer.manoeuvres import hohmann
from orbwatch.transfer.mission import (
    CircularOrbit,
    MissionAssumptions,
    RendezvousPlan,
    budget_items,
    direct_transfer,
    drift_options,
    local_time_of_ascending_node_h,
    orbit_from_tle,
    pareto_front,
    rideshare_orbit,
    terminal_approach,
)

MAX_TRACK_SAMPLES: int = 6000
MAX_TRACK_HALF_SPAN_S: float = 3.0 * SECONDS_PER_DAY
MAX_PASS_WINDOW_HOURS: float = 72.0
MIN_BEHAVIOUR_RECORDS: int = 20
"""Fewer element sets than this cannot give a meaningful noise estimate."""

ARCSEC_PER_RAD: float = math.degrees(1.0) * 3600.0

HELD_FRACTION: float = 0.3
"""A GEO orbit plane that moved less than this share of its free drift is held."""

DRIFT_MODEL_TOLERANCE: float = 0.1
"""Relative accuracy of the Laplace-plane drift model. It came within 9% of
INTELSAT 905's measured drift, so a containment requirement below 10% of the
cost of undoing the free drift is indistinguishable from no control at all."""


def _unix_ms(time: datetime) -> int:
    return int(round(time.timestamp() * 1000.0))


def _clean(values: np.ndarray, decimals: int) -> list[float | None]:
    """Round and replace NaN with None, since NaN is not valid JSON."""
    return [None if not math.isfinite(v) else round(float(v), decimals) for v in values]


def _number(value: float | None, decimals: int) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), decimals)


def nominal_period_min(tle: TLE) -> float:
    """Nominal period from the TLE mean motion, minutes.

    Fine for display and for sizing a plotting window. Not an osculating period,
    for the same reason the TLE elements are not osculating elements.
    """
    return 1440.0 / tle.mean_motion_rev_per_day


def object_summary(tle: TLE, now_utc: datetime) -> dict[str, Any]:
    """Identity, raw element set and mean elements for the Elements tab."""
    now = require_utc(now_utc)
    return {
        "norad_id": tle.norad_id,
        "name": tle.name or f"NORAD {tle.norad_id}",
        "international_designator": tle.international_designator,
        "classification": tle.classification,
        "epoch_utc": tle.epoch_utc.isoformat(),
        "epoch_unix_ms": _unix_ms(tle.epoch_utc),
        "age_days": round(tle.age_days(now), 4),
        "line1": tle.line1,
        "line2": tle.line2,
        "nominal_period_min": round(nominal_period_min(tle), 4),
        "mean_elements": {
            "inclination_deg": round(math.degrees(tle.inclination_rad), 4),
            "raan_deg": round(math.degrees(tle.raan_rad), 4),
            "eccentricity": tle.eccentricity,
            "argument_of_perigee_deg": round(math.degrees(tle.argp_rad), 4),
            "mean_anomaly_deg": round(math.degrees(tle.mean_anomaly_rad), 4),
            "mean_motion_rev_per_day": tle.mean_motion_rev_per_day,
            "bstar_per_earth_radius": tle.bstar_per_earth_radius,
            "revolution_number_at_epoch": tle.revolution_number,
            "element_set_number": tle.element_set_number,
        },
    }


def site_summary(site: GroundSite) -> dict[str, Any]:
    return {
        "name": site.name,
        "lat_deg": site.lat_deg,
        "lon_deg": site.lon_deg,
        "alt_km": site.alt_km,
    }


def track(
    tle: TLE,
    site: GroundSite,
    center_utc: datetime,
    before_s: float,
    after_s: float,
    step_s: float,
    min_elevation_deg: float = 0.0,
) -> dict[str, Any]:
    """Sampled ephemeris around an instant, with everything the globe draws.

    Parameters
    ----------
    tle : TLE
    site : GroundSite
    center_utc : datetime
        Simulation time the window is centred on.
    before_s, after_s : float
        Window extent either side of ``center_utc``, seconds.
    step_s : float
        Sample spacing, seconds.
    min_elevation_deg : float, optional
        Elevation mask used for the ``in_view`` flag, degrees.

    Returns
    -------
    dict
        Parallel arrays under ``samples``, one entry per instant. Geodetic
        coordinates on WGS-84, angles in degrees for display, distances in km.
    """
    center = require_utc(center_utc)
    if not 0.0 <= before_s <= MAX_TRACK_HALF_SPAN_S:
        raise ValueError(f"before_s must be in [0, {MAX_TRACK_HALF_SPAN_S:.0f}]")
    if not 0.0 <= after_s <= MAX_TRACK_HALF_SPAN_S:
        raise ValueError(f"after_s must be in [0, {MAX_TRACK_HALF_SPAN_S:.0f}]")
    if step_s < 1.0:
        raise ValueError("step_s must be at least 1 second")
    if (before_s + after_s) / step_s + 1 > MAX_TRACK_SAMPLES:
        raise ValueError(f"window would exceed {MAX_TRACK_SAMPLES} samples")

    times = time_grid(
        center - timedelta(seconds=before_s),
        center + timedelta(seconds=after_s),
        step_s,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", StaleElementSetWarning)
        ephemeris = propagate(tle, times, on_error="nan")
    stale = any(issubclass(w.category, StaleElementSetWarning) for w in caught)

    r_ecef, _ = teme_to_ecef(ephemeris.r_teme_km, ephemeris.v_teme_km_s, times)
    lat, lon, alt = ecef_to_geodetic(r_ecef)
    azimuth, elevation, range_km = look_angles(
        r_ecef, site.lat_rad, site.lon_rad, site.alt_km
    )
    speed = np.linalg.norm(ephemeris.v_teme_km_s, axis=1)

    sun_teme = sun_position_km(times)
    sunlit = is_sunlit(ephemeris.r_teme_km, sun_teme)
    sun_ecef = teme_to_ecef_position(sun_teme, times)
    subsolar_lat, subsolar_lon = subsolar_point_rad(sun_ecef)
    _, site_sun_elevation, _ = look_angles(
        sun_ecef, site.lat_rad, site.lon_rad, site.alt_km
    )

    mask_rad = math.radians(min_elevation_deg)
    valid = ephemeris.valid
    in_view = valid & (elevation > mask_rad)
    optically_visible = (
        in_view & sunlit & (site_sun_elevation < CIVIL_TWILIGHT_SUN_ELEVATION_RAD)
    )

    age_days = ephemeris.ages_days
    orbit_number = tle.revolution_number + age_days * tle.mean_motion_rev_per_day

    return {
        "norad_id": tle.norad_id,
        "frame": "WGS-84 geodetic from TEME via GMST rotation",
        "site": site_summary(site),
        "stale": stale,
        "samples": {
            "t_unix_ms": [_unix_ms(t) for t in times],
            "valid": valid.tolist(),
            "lat_deg": _clean(np.rad2deg(lat), 5),
            "lon_deg": _clean(np.rad2deg(lon), 5),
            "alt_km": _clean(alt, 3),
            "speed_km_s": _clean(speed, 5),
            "footprint_deg": _clean(np.rad2deg(horizon_half_angle_rad(alt)), 4),
            "azimuth_deg": _clean(np.rad2deg(azimuth), 3),
            "elevation_deg": _clean(np.rad2deg(elevation), 3),
            "range_km": _clean(range_km, 2),
            "sunlit": (sunlit & valid).tolist(),
            "in_view": in_view.tolist(),
            "optically_visible": optically_visible.tolist(),
            "subsolar_lat_deg": _clean(np.rad2deg(subsolar_lat), 4),
            "subsolar_lon_deg": _clean(np.rad2deg(subsolar_lon), 4),
            "site_sun_elevation_deg": _clean(np.rad2deg(site_sun_elevation), 3),
            "age_days": _clean(age_days, 5),
            "orbit_number": _clean(orbit_number, 3),
        },
    }


def passes(
    tle: TLE,
    site: GroundSite,
    start_utc: datetime,
    hours: float,
    min_elevation_deg: float = 0.0,
) -> dict[str, Any]:
    """Upcoming passes over a site, for the Passes tab."""
    start = require_utc(start_utc)
    if not 0.0 < hours <= MAX_PASS_WINDOW_HOURS:
        raise ValueError(f"hours must be in (0, {MAX_PASS_WINDOW_HOURS:.0f}]")
    stop = start + timedelta(hours=hours)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", StaleElementSetWarning)
        found = find_passes(tle, site, start, stop, min_elevation_deg=min_elevation_deg)
    stale = any(issubclass(w.category, StaleElementSetWarning) for w in caught)

    return {
        "norad_id": tle.norad_id,
        "site": site_summary(site),
        "window_start_unix_ms": _unix_ms(start),
        "window_stop_unix_ms": _unix_ms(stop),
        "min_elevation_deg": min_elevation_deg,
        "stale": stale,
        "passes": [
            {
                "aos_unix_ms": _unix_ms(p.aos_utc),
                "los_unix_ms": _unix_ms(p.los_utc),
                "max_elevation_unix_ms": _unix_ms(p.max_elevation_utc),
                "max_elevation_deg": round(math.degrees(p.max_elevation_rad), 2),
                "aos_azimuth_deg": round(math.degrees(p.aos_azimuth_rad), 1),
                "los_azimuth_deg": round(math.degrees(p.los_azimuth_rad), 1),
                "duration_s": round(p.duration_s, 1),
                "starts_before_window": p.starts_before_window,
                "ends_after_window": p.ends_after_window,
            }
            for p in found
        ],
    }


# --------------------------------------------------------------------------
# Behaviour: the events layer, for the interface's history view
# --------------------------------------------------------------------------


def _event(event: Manoeuvre, status: str) -> dict[str, Any]:
    """One detected or set-aside event. Out-of-plane change is in arcseconds,
    signed for a scalar inclination and a magnitude for the GEO vector."""
    if event.kind == "in_plane":
        change = {"delta_a_km": _number(event.change[0], 4)}
    elif len(event.change) == 1:
        change = {"change_arcsec": _number(event.change[0] * ARCSEC_PER_RAD, 2)}
    else:
        size = float(np.linalg.norm(event.change)) * ARCSEC_PER_RAD
        change = {"change_arcsec": _number(size, 2)}
    return {
        "status": status,
        "kind": event.kind,
        "profile": event.profile,
        "start_unix_ms": _unix_ms(event.start_utc),
        "end_unix_ms": _unix_ms(event.end_utc),
        "window_hours": _number(event.window_hours, 2),
        "delta_v_m_s": _number(event.delta_v_m_s, 4),
        "significance": _number(event.significance, 1),
        "coherence": _number(event.coherence, 3),
        "gaps": event.gaps,
        **change,
    }


def scatter_per_gap_m_s(history: OrbitHistory, analysis: StepAnalysis) -> float:
    """The detector's per-gap scatter, one robust sigma, as a speed change.

    Uses the same conversions as the manoeuvre delta-v: (v/2) |da| / a in
    plane, v |di| out of plane. This is what a correction has to stand out
    from, gap by gap. It is called scatter rather than noise because it is not
    only measurement error: control applied more often than the catalogue
    updates lands in it too. INMARSAT 5-F3, kept by daily ion thruster firings,
    scatters 60 times more in semi-major axis than INTELSAT 905, which coasts
    between burns weeks apart.
    """
    v_m_s = history.mean_speed_km_s * 1000.0
    if analysis.kind == "in_plane":
        return 0.5 * v_m_s * analysis.sigma / float(np.mean(history.a_km))
    return v_m_s * analysis.sigma


def _budget(
    budget: Budget,
    history: OrbitHistory,
    analysis: StepAnalysis,
    manoeuvres: Sequence[Manoeuvre],
) -> dict[str, Any]:
    """A budget, plus the comparison that says whether its corrections could
    be seen one at a time: their typical size against the per-gap scatter.
    Burns are impulsive manoeuvres only; a sustained event such as a drag surge
    is not a burn."""
    burns = [
        m.delta_v_m_s
        for m in manoeuvres
        if m.kind == budget.kind and m.profile == "impulsive"
    ]
    gaps = int(analysis.usable.sum())
    required = budget.required_m_s
    return {
        "detected_m_s": _number(budget.detected_m_s, 3),
        "detected_per_year_m_s": _number(budget.per_year(budget.detected_m_s), 3),
        "required_m_s": _number(required, 3),
        "required_per_year_m_s": _number(budget.per_year(required), 3),
        "closure": _number(budget.closure, 4),
        "method": budget.method,
        "burns": len(burns),
        "typical_burn_m_s": _number(float(np.median(burns)), 4) if burns else None,
        "scatter_per_gap_m_s": _number(scatter_per_gap_m_s(history, analysis), 5),
        "required_per_gap_m_s": (
            _number(required / gaps, 5) if required and gaps else None
        ),
    }


def plane_control(
    natural_displacement_rad: float,
    observed_displacement_rad: float,
    required_m_s: float | None,
    speed_m_s: float,
) -> str:
    """Whether a GEO orbit plane was held, left free, or something between.

    ``held`` when it moved less than ``HELD_FRACTION`` of its free drift.
    ``free`` when the containment requirement is within the drift model's
    accuracy of the cost of undoing the free drift, so it is indistinguishable
    from no control. ``partial`` otherwise.
    """
    if observed_displacement_rad < HELD_FRACTION * natural_displacement_rad:
        return "held"
    free_drift_cost = speed_m_s * natural_displacement_rad
    if required_m_s is not None and required_m_s <= DRIFT_MODEL_TOLERANCE * (
        free_drift_cost
    ):
        return "free"
    return "partial"


def _detector(
    analysis: StepAnalysis, t_days: np.ndarray, first: datetime, scale: float
) -> dict[str, Any]:
    """What the detector compares for each usable gap: the size of the change
    nature cannot explain, against the threshold that flags it. Edge gaps carry
    their own raised threshold in ``edge_threshold``, null elsewhere."""
    usable = analysis.usable
    midpoint_days = 0.5 * (t_days[1:] + t_days[:-1])[usable]
    edge_threshold = np.where(analysis.edge, analysis.gap_threshold, np.nan)
    return {
        "unit": "m" if analysis.kind == "in_plane" else "arcsec",
        "t_unix_ms": [
            _unix_ms(first + timedelta(days=float(d))) for d in midpoint_days
        ],
        "magnitude": _clean(analysis.magnitude[usable] * scale, 3),
        "flagged": analysis.flagged[usable].tolist(),
        "edge_threshold": _clean(edge_threshold[usable] * scale, 3),
        "sigma": _number(analysis.sigma * scale, 4),
        "threshold": _number(analysis.threshold * scale, 4),
    }


def _natural_drift(report: EventReport) -> dict[str, Any] | None:
    """For GEO: where the orbit plane would have gone with nobody steering it,
    from the observed starting state, next to where it actually went."""
    h = report.history
    if h.regime != "geosynchronous":
        return None
    t, ivec = h.t_days, h.inclination_vector_rad
    start = np.median(ivec[t <= t[0] + ENDPOINT_WINDOW_DAYS], axis=0)
    end = np.median(ivec[t >= t[-1] - ENDPOINT_WINDOW_DAYS], axis=0)
    days = np.linspace(0.0, h.span_days, max(2, int(h.span_days) + 1))
    path_rad = natural_inclination_path_rad(start, days)
    path = np.rad2deg(path_rad)
    natural = float(np.linalg.norm(path_rad[-1] - path_rad[0]))
    observed = float(np.linalg.norm(end - start))
    speed_m_s = h.mean_speed_km_s * 1000.0
    return {
        "control": plane_control(
            natural, observed, report.out_of_plane_budget.required_m_s, speed_m_s
        ),
        "free_drift_cost_m_s": _number(speed_m_s * natural, 3),
        "t_unix_ms": [
            _unix_ms(h.epochs_utc[0] + timedelta(days=float(d))) for d in days
        ],
        "ivec_x_deg": _clean(path[:, 0], 6),
        "ivec_y_deg": _clean(path[:, 1], 6),
        "natural_displacement_deg": _number(math.degrees(natural), 5),
        "observed_displacement_deg": _number(math.degrees(observed), 5),
    }


def _score(score: Score) -> dict[str, Any]:
    return {
        "median_abs_days": _number(score.median_abs_days, 3),
        "mean_abs_days": _number(score.mean_abs_days, 3),
        "within_one_day": _number(score.within_one_day, 3),
        "bias_days": _number(score.bias_days, 3),
    }


def _pattern(report: EventReport) -> dict[str, Any] | None:
    """East-west pattern of life at GEO: the policy, the forecast, its track
    record, and the series and fitted arc the interface draws."""
    pattern = pattern_of_life(report.history, report.manoeuvres)
    if pattern is None:
        return None
    if not pattern.applies:
        return {
            "applies": False,
            "reason": pattern.reason,
            "burn_count": pattern.burns,
            "weekday_counts": list(pattern.weekday_counts),
        }
    h = report.history
    p = pattern.policy
    lon = longitude_deg(h)

    arc = None
    if pattern.trigger_forecast.arc is not None:
        start, lon0, rate, accel = pattern.trigger_forecast.arc
        end = max(pattern.trigger_forecast.predicted_utc, h.epochs_utc[-1])
        tau = np.linspace(0.0, (end - start).total_seconds() / 86400.0, 80)
        arc = {
            "t_unix_ms": [_unix_ms(start + timedelta(days=float(x))) for x in tau],
            "lon_deg": _clean(lon0 + rate * tau + 0.5 * accel * tau**2, 5),
        }

    preferred_error = None
    if pattern.backtests:
        preferred_error = pattern.backtests[0].score(pattern.preferred).median_abs_days
    return {
        "applies": True,
        "reason": "",
        "burn_count": pattern.burns,
        "policy": {
            "burns": p.burns,
            "interval_days": _number(p.interval_days, 3),
            "interval_spread_days": _number(p.interval_spread_days, 3),
            "delta_v_m_s": _number(p.delta_v_m_s, 4),
            "delta_a_km": _number(p.delta_a_km, 4),
            "trigger_longitude_deg": _number(p.trigger_longitude_deg, 5),
            "trigger_spread_deg": _number(p.trigger_spread_deg, 5),
            "acceleration_deg_per_day2": _number(p.acceleration_deg_per_day2, 6),
            "model_acceleration_deg_per_day2": _number(
                p.model_acceleration_deg_per_day2, 6
            ),
            "box_deg": [_number(p.box_deg[0], 5), _number(p.box_deg[1], 5)],
            "edge_drift_deg_per_day": _number(p.edge_drift_deg_per_day, 5),
            "timing_floor_days": _number(p.timing_floor_days, 3),
        },
        "preferred": pattern.preferred,
        "forecast_unix_ms": _unix_ms(pattern.forecast.predicted_utc),
        "forecast_error_days": _number(preferred_error, 3),
        "trigger_forecast_unix_ms": _unix_ms(pattern.trigger_forecast.predicted_utc),
        "trigger_fell_back": pattern.trigger_forecast.fell_back,
        "data_end_unix_ms": _unix_ms(h.epochs_utc[-1]),
        "backtests": [
            {
                "lead_days": b.lead_days,
                "burns": len(b.entries),
                "skipped": b.skipped,
                "truth_half_width_days": _number(b.truth_half_width_days, 3),
                "interval": _score(b.score("interval")),
                "trigger": _score(b.score("trigger")),
            }
            for b in pattern.backtests
        ],
        "weekday_counts": list(pattern.weekday_counts),
        "longitude": {
            "t_unix_ms": [_unix_ms(t) for t in h.epochs_utc],
            "lon_deg": _clean(lon, 5),
        },
        "burns": [
            [_unix_ms(b.start_utc), _unix_ms(b.end_utc)]
            for b in east_west_burns(report.manoeuvres)
        ],
        "arc": arc,
    }


def behaviour(records: Sequence[TLE], sigmas: float = DEFAULT_SIGMAS) -> dict[str, Any]:
    """An object's history, manoeuvres, set-aside events and budgets.

    Parameters
    ----------
    records : sequence of TLE
        Element set history for one object, e.g. from Space-Track.
    sigmas : float, optional
        Detection threshold in robust standard deviations.

    Returns
    -------
    dict
        ``series`` holds the cleaned history as parallel arrays. ``detector``
        holds, per signal, the per-gap change nature cannot explain and the
        threshold it is compared with. ``events`` lists manoeuvres and, marked by
        ``status``, the events set aside and why. ``natural`` is the free drift
        of the orbit plane at GEO and null elsewhere.

    Raises
    ------
    ValueError
        With fewer than ``MIN_BEHAVIOUR_RECORDS`` element sets.
    """
    if len(records) < MIN_BEHAVIOUR_RECORDS:
        raise ValueError(
            f"only {len(records)} element sets in the window; at least "
            f"{MIN_BEHAVIOUR_RECORDS} are needed to separate burns from noise"
        )
    report = analyse(records, sigmas=sigmas)
    h = report.history
    q = h.quality
    ivec_deg = np.rad2deg(h.inclination_vector_rad)

    events = [_event(m, "manoeuvre") for m in report.manoeuvres]
    for status, group in (
        ("drag_surge", report.drag_surges),
        ("below_floor", report.below_floor),
        ("incoherent", report.incoherent),
        ("excursion", report.excursions),
    ):
        events += [_event(m, status) for m in group]
    events.sort(key=lambda e: e["start_unix_ms"])

    return {
        "norad_id": h.norad_id,
        "regime": h.regime,
        "start_unix_ms": _unix_ms(h.epochs_utc[0]),
        "stop_unix_ms": _unix_ms(h.epochs_utc[-1]),
        "span_days": _number(h.span_days, 3),
        "sigmas": sigmas,
        "quality": {
            "records_in": q.records_in,
            "refits_merged": q.refits_merged,
            "transients_rejected": q.transients_rejected,
            "used": len(h),
        },
        "series": {
            "t_unix_ms": [_unix_ms(t) for t in h.epochs_utc],
            "a_km": _clean(h.a_km, 4),
            "inclination_deg": _clean(np.rad2deg(h.inclination_rad), 6),
            "ivec_x_deg": _clean(ivec_deg[:, 0], 6),
            "ivec_y_deg": _clean(ivec_deg[:, 1], 6),
        },
        "rejected_unix_ms": [_unix_ms(t) for t in q.transient_epochs_utc],
        "detector": {
            "in_plane": _detector(report.in_plane, h.t_days, h.epochs_utc[0], 1000.0),
            "out_of_plane": _detector(
                report.out_of_plane, h.t_days, h.epochs_utc[0], ARCSEC_PER_RAD
            ),
        },
        "floors_m_s": {
            "in_plane": floor_m_s("in_plane", h.regime),
            "out_of_plane": floor_m_s("out_of_plane", h.regime),
        },
        "floor_override_sigmas": dict(FLOOR_OVERRIDE_SIGMAS),
        "events": events,
        "budgets": {
            "in_plane": _budget(
                report.in_plane_budget, h, report.in_plane, report.manoeuvres
            ),
            "out_of_plane": _budget(
                report.out_of_plane_budget, h, report.out_of_plane, report.manoeuvres
            ),
        },
        "natural": _natural_drift(report),
        "pattern": _pattern(report),
    }


# --------------------------------------------------------------------------
# Mission: rendezvous planning, for the interface's mission view
# --------------------------------------------------------------------------

MISSION_MAX_TARGET_ALTITUDE_KM: float = 2000.0
"""The planner starts from a sun-synchronous rideshare and uses J2 to line up
planes, which only makes sense for low-orbit targets."""

MISSION_SHOWN_DAYS: float = 730.0
MISSION_DEFAULT_BUDGET_DAYS: float = 180.0
PROXIMITY_RUNS: int = 1000
PROXIMITY_SHOWN_RUNS: int = 20
PROXIMITY_STEP_S: float = 120.0
"""Sampling of the drawn trajectories. The Monte Carlo itself checks the
keep-out sphere every 60 s."""


def _path(states: np.ndarray) -> list[list[float]]:
    """Radial, in-track, cross-track positions rounded to 0.1 m for drawing."""
    return np.round(states[..., :3], 1).tolist()


def _proximity(plan: ProximityPlan, error_scale: float) -> dict[str, Any]:
    """The designed approach, what each burn failing would do, and the Monte
    Carlo that sizes the budget line."""
    d, mc = plan.design, plan.monte_carlo
    checks = {c.burn: c for c in plan.checks}
    stride = max(1, round(PROXIMITY_STEP_S / 60.0))
    burns = []
    for k, burn in enumerate(d.burns):
        entry: dict[str, Any] = {
            "label": burn.label,
            "kind": burn.kind,
            "time_h": _number(burn.time_s / 3600.0, 4),
            "dv_m_s": _number(burn.dv_m_s, 4),
            "p99_m_s": _number(np.percentile(mc.burn_dv_m_s[:, k], 99), 4),
            "position_m": _path(burn.state_before),
            "missed": None,
        }
        check = checks.get(k)
        if check is not None:
            coast = np.arange(0.0, d.safety_orbits * d.period_s, PROXIMITY_STEP_S)
            entry["missed"] = {
                "closest_m": _number(check.min_distance_m, 1),
                "after_h": _number(check.time_after_s / 3600.0, 3),
                "passes": check.passes,
                "path": _path(cw.propagate(burn.state_before, d.n, coast)),
            }
        burns.append(entry)
    times, nominal = d.trajectory(PROXIMITY_STEP_S)
    base = Dispersions()
    return {
        "hold_m": _number(-d.burns[0].state_before[1], 1),
        "ellipse_radial_m": d.ellipse_radial_m,
        "ellipse_cross_track_m": d.ellipse_cross_track_m,
        "keep_out_m": d.keep_out_m,
        "inspection_orbits": d.inspection_orbits,
        "safety_orbits": d.safety_orbits,
        "period_min": _number(d.period_s / 60.0, 2),
        "duration_h": _number(d.end_time_s / 3600.0, 3),
        "dv_m_s": _number(d.dv_m_s, 3),
        "all_safe": all(c.passes for c in plan.checks),
        "burns": burns,
        "nominal": _path(nominal),
        "nominal_h": _clean(times / 3600.0, 4),
        "monte_carlo": {
            "runs": mc.runs,
            "percentile": PROXIMITY_PERCENTILE,
            "dv_mean_m_s": _number(float(mc.dv_m_s.mean()), 3),
            "dv_budget_m_s": _number(plan.budget_dv_m_s, 3),
            "closest_p1_m": _number(float(np.percentile(mc.min_distance_m, 1)), 1),
            "inside_keep_out": _number(mc.violation_fraction, 4),
            "dv_m_s": _clean(mc.dv_m_s, 3),
            "closest_m": _clean(mc.min_distance_m, 1),
            "shown": _path(mc.sample_states[:PROXIMITY_SHOWN_RUNS, ::stride]),
            "shown_h": _clean(mc.sample_times_s[::stride] / 3600.0, 4),
            "error_scale": error_scale,
            "errors": {
                "range_percent": 100 * base.range_fraction * error_scale,
                "bearing_deg": base.bearing_deg * error_scale,
                "burn_magnitude_percent": 100 * base.magnitude_fraction * error_scale,
                "burn_pointing_deg": base.pointing_deg * error_scale,
                "arrival_m": base.arrival_position_m * error_scale,
            },
        },
    }


def mission(
    tle: TLE,
    epoch_utc: datetime,
    dropoff_altitude_km: float = 525.0,
    node_offset_deg: float = -15.0,
    dry_mass_kg: float = 150.0,
    isp_s: float = 220.0,
    assumptions: MissionAssumptions | None = None,
    ellipse_m: float = ELLIPSE_RADIAL_M,
    keep_out_m: float = KEEP_OUT_M,
    inspection_orbits: int = INSPECTION_ORBITS,
    error_scale: float = 1.0,
) -> dict[str, Any]:
    """The whole rendezvous trade for one target, with a budget for every option.

    Every option on the time against delta-v front comes with its complete
    budget, so the interface can move along the front without asking again.
    The proximity operations are designed and Monte Carlo'd once, around the
    target, and their 99th-percentile delta-v is the proximity budget line.
    ``ellipse_m`` sizes the safety ellipse radially and cross-track alike, and
    ``error_scale`` multiplies every navigation and execution error.

    Raises
    ------
    ValueError
        For targets above ``MISSION_MAX_TARGET_ALTITUDE_KM``, or a safety
        ellipse no wider than the keep-out sphere.
    """
    assumptions = assumptions or MissionAssumptions()
    epoch = require_utc(epoch_utc)
    target = orbit_from_tle(tle)
    if target.altitude_km > MISSION_MAX_TARGET_ALTITUDE_KM:
        raise ValueError(
            f"{tle.name or 'This object'} is at {target.altitude_km:,.0f} km. Mission"
            " planning starts from a sun-synchronous rideshare and uses J2 to line up"
            " orbit planes, which only works for low-orbit targets."
        )
    target_now = target.at(epoch)
    ltan_target = local_time_of_ascending_node_h(target, epoch)
    dropoff, dropoff_kind = rideshare_orbit(
        target, dropoff_altitude_km, np.deg2rad(node_offset_deg), epoch
    )
    front = [
        o
        for o in pareto_front(drift_options(dropoff, target))
        if o.total_days <= MISSION_SHOWN_DAYS
    ]
    if not front:
        raise ValueError("no drift orbit lines up the planes within two years")
    approach = terminal_approach(target_now)
    proximity = plan_proximity(
        target.a_km,
        Dispersions().scaled(error_scale),
        runs=PROXIMITY_RUNS,
        hold_m=approach.hold_km * 1000.0,
        ellipse_radial_m=ellipse_m,
        ellipse_cross_track_m=ellipse_m,
        keep_out_m=keep_out_m,
        inspection_orbits=int(inspection_orbits),
    )
    proximity_line = (
        proximity.budget_dv_m_s,
        f"Monte Carlo {PROXIMITY_PERCENTILE:g}th percentile, {PROXIMITY_RUNS} runs",
    )
    direct = direct_transfer(dropoff, target)
    delta_i = target.inclination_rad - dropoff.inclination_rad
    floor = hohmann(dropoff.a_km, target.a_km, delta_i)
    gap = float(np.rad2deg(_wrap(target_now.raan_rad - dropoff.raan_rad)))

    options = []
    labels: list[str] = []
    categories: list[str] = []
    for option in front:
        plan = RendezvousPlan(
            dropoff=dropoff,
            target=target_now,
            delta_raan_rad=np.deg2rad(gap),
            delta_inclination_rad=delta_i,
            max_days=option.total_days,
            chosen=option,
            front=(),
            direct=direct,
            approach=approach,
        )
        items = budget_items(plan, assumptions, proximity_line)
        budget = build_budget(items, dry_mass_kg, isp_s)
        labels = [i.label for i in items]
        categories = [i.category for i in items]
        drift = CircularOrbit(
            EARTH_RADIUS_KM + option.altitude_km,
            option.inclination_rad,
            dropoff.raan_rad,
            epoch,
        )
        closing = np.rad2deg(target.nodal_rate_rad_s - drift.nodal_rate_rad_s) * 86400
        options.append(
            {
                "total_days": _number(option.total_days, 3),
                "wait_days": _number(option.wait_days, 3),
                "phasing_days": _number(option.phasing_days, 3),
                "altitude_km": _number(option.altitude_km, 1),
                "inclination_deg": _number(np.rad2deg(option.inclination_rad), 4),
                "enter_m_s": _number(option.enter.dv_km_s * 1000, 2),
                "leave_m_s": _number(option.leave.dv_km_s * 1000, 2),
                "transfer_m_s": _number(option.dv_km_s * 1000, 2),
                "gap_rate_deg_per_day": _number(closing, 6),
                "arrival_unix_ms": _unix_ms(plan.arrival_utc),
                "lines": [
                    [_number(i.dv_m_s, 2), _number(i.dv_with_margin_m_s, 2)]
                    for i in items
                ],
                "basis": [i.basis for i in items],
                "dv_nominal_m_s": _number(budget.dv_nominal_m_s, 2),
                "dv_margined_m_s": _number(budget.dv_with_margins_m_s, 2),
                "propellant_kg": _number(budget.propellant_kg, 2),
                "wet_mass_kg": _number(budget.wet_mass_kg, 2),
            }
        )
    dry_margined = build_budget([], dry_mass_kg, isp_s).dry_mass_kg
    return {
        "target": {
            "norad_id": tle.norad_id,
            "name": tle.name or f"NORAD {tle.norad_id}",
            "altitude_km": _number(target.altitude_km, 2),
            "inclination_deg": _number(np.rad2deg(target.inclination_rad), 4),
            "ltan_h": _number(ltan_target, 4),
        },
        "dropoff": {
            "altitude_km": _number(dropoff.altitude_km, 2),
            "inclination_deg": _number(np.rad2deg(dropoff.inclination_rad), 4),
            "ltan_h": _number(local_time_of_ascending_node_h(dropoff, epoch), 4),
            "node_offset_deg": node_offset_deg,
            "kind": dropoff_kind,
        },
        "epoch_unix_ms": _unix_ms(epoch),
        "gap": {
            "raan_deg": _number(gap, 4),
            "inclination_deg": _number(np.rad2deg(delta_i), 4),
        },
        "direct_m_s": _number(direct.dv_km_s * 1000, 1),
        "floor_m_s": _number(floor.dv_km_s * 1000, 2),
        "approach": {
            "dv_m_s": _number(approach.dv_km_s * 1000, 3),
            "minutes": _number(approach.time_of_flight_s / 60, 1),
            "far_km": approach.far_km,
            "hold_km": approach.hold_km,
        },
        "proximity": _proximity(proximity, error_scale),
        "lines": [
            {
                "label": label,
                "category": category,
                "margin": MARGINS[category],
                "requirement": REQUIREMENTS[category],
            }
            for label, category in zip(labels, categories, strict=True)
        ],
        "options": options,
        "default_budget_days": MISSION_DEFAULT_BUDGET_DAYS,
        "spacecraft": {
            "dry_mass_kg": dry_mass_kg,
            "dry_mass_with_margin_kg": _number(dry_margined, 2),
            "isp_s": isp_s,
        },
        "assumptions": {
            "injection_altitude_error_km": assumptions.injection_altitude_error_km,
            "injection_inclination_error_deg": (
                assumptions.injection_inclination_error_deg
            ),
            "operations_days": assumptions.operations_days,
            "disposal_perigee_km": assumptions.disposal_perigee_km,
        },
        "reference": BUDGET_REFERENCE,
        "model": "mean J2 dynamics, circular orbits, impulsive burns, no drag",
    }


def _wrap(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)
