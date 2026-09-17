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
from orbwatch.catalog.frames import (
    ecef_to_geodetic,
    look_angles,
    teme_to_ecef,
    teme_to_ecef_position,
)
from orbwatch.catalog.propagate import StaleElementSetWarning, propagate
from orbwatch.catalog.timescales import SECONDS_PER_DAY, require_utc, time_grid
from orbwatch.catalog.tle import TLE

MAX_TRACK_SAMPLES: int = 6000
MAX_TRACK_HALF_SPAN_S: float = 3.0 * SECONDS_PER_DAY
MAX_PASS_WINDOW_HOURS: float = 72.0


def _unix_ms(time: datetime) -> int:
    return int(round(time.timestamp() * 1000.0))


def _clean(values: np.ndarray, decimals: int) -> list[float | None]:
    """Round and replace NaN with None, since NaN is not valid JSON."""
    return [None if not math.isfinite(v) else round(float(v), decimals) for v in values]


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
