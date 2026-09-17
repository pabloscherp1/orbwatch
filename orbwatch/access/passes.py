"""Ground station visibility: look angles over time and pass prediction.

A pass is an interval during which an object stands above a ground site's
elevation mask. AOS (acquisition of signal) is when it rises through the mask
and LOS (loss of signal) is when it sets.

Units: km and radians internally. :class:`GroundSite` takes degrees because
that is how every site coordinate in the world is published.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import ArrayLike, NDArray

from orbwatch.catalog.frames import look_angles, teme_to_ecef_position
from orbwatch.catalog.propagate import StaleElementSetWarning, propagate
from orbwatch.catalog.timescales import require_utc, time_grid
from orbwatch.catalog.tle import TLE

MEAN_EARTH_RADIUS_KM: float = 6371.0088
"""IUGG mean Earth radius, km. Used only for spherical coverage geometry."""


@dataclass(frozen=True)
class GroundSite:
    """A fixed observing site on the WGS-84 ellipsoid.

    Attributes
    ----------
    name : str
    lat_deg : float
        Geodetic latitude, degrees, in [-90, 90].
    lon_deg : float
        Longitude, degrees east, in [-180, 360).
    alt_km : float
        Height above the ellipsoid, km.
    """

    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError(f"latitude {self.lat_deg} deg is outside [-90, 90]")
        if not -180.0 <= self.lon_deg < 360.0:
            raise ValueError(f"longitude {self.lon_deg} deg is outside [-180, 360)")
        if not -0.5 <= self.alt_km <= 10.0:
            raise ValueError(f"site altitude {self.alt_km} km is not a ground site")

    @property
    def lat_rad(self) -> float:
        return float(np.deg2rad(self.lat_deg))

    @property
    def lon_rad(self) -> float:
        return float(np.deg2rad(self.lon_deg))


@dataclass(frozen=True)
class Pass:
    """One interval of visibility above a site's elevation mask.

    ``starts_before_window`` and ``ends_after_window`` flag passes that were
    already in progress at the start of the search, or still in progress at
    its end. For those, the clipped AOS or LOS is the window edge, not a real
    horizon crossing, and a geostationary object seen from under it is simply
    one pass clipped at both ends.
    """

    aos_utc: datetime
    los_utc: datetime
    max_elevation_utc: datetime
    max_elevation_rad: float
    aos_azimuth_rad: float
    los_azimuth_rad: float
    starts_before_window: bool
    ends_after_window: bool

    @property
    def duration_s(self) -> float:
        return (self.los_utc - self.aos_utc).total_seconds()


def horizon_half_angle_rad(
    alt_km: ArrayLike, min_elevation_rad: float = 0.0
) -> NDArray:
    """Earth central angle from the sub-satellite point to the edge of coverage.

    Parameters
    ----------
    alt_km : array_like
        Object altitude, km.
    min_elevation_rad : float, optional
        Elevation mask at the edge of coverage, radians.

    Returns
    -------
    ndarray
        Half-angle of the coverage circle on the Earth's surface, radians. A
        ground site closer than this to the sub-satellite point sees the object
        above the mask.

    Notes
    -----
    Spherical Earth of mean radius. In the triangle formed by the Earth's
    centre, the site and the object, the angle at the site is 90 deg plus the
    mask, which gives

        lambda = arccos( R cos(mask) / (R + h) ) - mask.

    With no mask this is the familiar arccos(R / (R + h)), about 20 deg for the
    ISS and about 81 deg for a geostationary object.
    """
    alt = np.asarray(alt_km, dtype=float)
    ratio = (
        MEAN_EARTH_RADIUS_KM * np.cos(min_elevation_rad) / (MEAN_EARTH_RADIUS_KM + alt)
    )
    return np.arccos(np.clip(ratio, -1.0, 1.0)) - min_elevation_rad


def look_angles_over_time(
    tle: TLE,
    site: GroundSite,
    times_utc: Sequence[datetime],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Azimuth, elevation and range from a site, for an element set over time.

    Samples where SGP4 fails come back as NaN in all three outputs.
    """
    ephemeris = propagate(tle, times_utc, on_error="nan")
    r_ecef = teme_to_ecef_position(ephemeris.r_teme_km, ephemeris.times_utc)
    return look_angles(r_ecef, site.lat_rad, site.lon_rad, site.alt_km)


def _elevation_rad(tle: TLE, site: GroundSite, time: datetime) -> float:
    _, elevation, _ = look_angles_over_time(tle, site, [time])
    return float(elevation[0])


def _azimuth_rad(tle: TLE, site: GroundSite, time: datetime) -> float:
    azimuth, _, _ = look_angles_over_time(tle, site, [time])
    return float(azimuth[0])


def _refine_crossing(
    tle: TLE,
    site: GroundSite,
    below: datetime,
    above: datetime,
    mask_rad: float,
    tolerance_s: float,
) -> datetime:
    """Bisect for the instant the elevation crosses the mask.

    ``below`` and ``above`` bracket the crossing and may be in either time
    order, so the same routine serves rises and sets.
    """
    while abs((above - below).total_seconds()) > tolerance_s:
        middle = below + (above - below) / 2
        elevation = _elevation_rad(tle, site, middle)
        if np.isfinite(elevation) and elevation > mask_rad:
            above = middle
        else:
            below = middle
    return above


def _refine_maximum(
    tle: TLE,
    site: GroundSite,
    start: datetime,
    stop: datetime,
    tolerance_s: float,
) -> tuple[datetime, float]:
    """Culmination time and elevation within a pass.

    Samples the pass every few seconds to find the neighbourhood of the peak,
    then golden-section searches around it. Elevation is single-peaked within
    a pass for any realistic geometry, which is what golden-section needs.
    """
    span_s = (stop - start).total_seconds()
    if span_s <= 0.0:
        return start, _elevation_rad(tle, site, start)

    step_s = min(5.0, span_s)
    times = list(time_grid(start, stop, step_s))
    if times[-1] < stop:
        times.append(stop)
    _, elevation, _ = look_angles_over_time(tle, site, times)
    best = int(np.nanargmax(elevation))

    low = max(start, times[best] - timedelta(seconds=step_s))
    high = min(stop, times[best] + timedelta(seconds=step_s))
    inverse_golden = (np.sqrt(5.0) - 1.0) / 2.0

    while (high - low).total_seconds() > tolerance_s:
        width = high - low
        left = high - width * inverse_golden
        right = low + width * inverse_golden
        if _elevation_rad(tle, site, left) < _elevation_rad(tle, site, right):
            low = left
        else:
            high = right

    peak = low + (high - low) / 2
    return peak, _elevation_rad(tle, site, peak)


def find_passes(
    tle: TLE,
    site: GroundSite,
    start_utc: datetime,
    stop_utc: datetime,
    min_elevation_deg: float = 0.0,
    coarse_step_s: float = 30.0,
    time_tolerance_s: float = 0.1,
) -> list[Pass]:
    """Predict passes of an object over a ground site.

    Parameters
    ----------
    tle : TLE
        Element set of the object.
    site : GroundSite
        Observing site.
    start_utc, stop_utc : datetime
        Search window, timezone-aware.
    min_elevation_deg : float, optional
        Elevation mask, degrees. Real sites rarely see down to 0 deg because of
        terrain, buildings and atmospheric losses; 5 to 10 deg is typical.
    coarse_step_s : float, optional
        Spacing of the initial scan, seconds.
    time_tolerance_s : float, optional
        Precision to which AOS, LOS and culmination are refined, seconds.

    Returns
    -------
    list of Pass
        In time order.

    Notes
    -----
    Scan coarsely for sign changes of elevation minus mask, bisect each change
    to ``time_tolerance_s``, then golden-section search for the peak.

    **Known blind spot.** A pass shorter than ``coarse_step_s``, or a dip below
    the mask and back within one step, can be missed entirely, because the scan
    never samples it. At 30 s this only affects grazing passes a few tenths of a
    degree above a mask, but it is a real limitation, not a rounding error.

    **The result is only as good as the element set.** Pass times inherit the
    along-track error of the TLE, which grows by kilometres per day, and at
    orbital speed a kilometre along-track is a fraction of a second of timing.
    """
    start = require_utc(start_utc)
    stop = require_utc(stop_utc)
    if stop <= start:
        raise ValueError("stop_utc must be after start_utc")
    mask_rad = float(np.deg2rad(min_elevation_deg))

    grid = list(time_grid(start, stop, coarse_step_s))
    if grid[-1] < stop:
        grid.append(stop)

    # Let the stale element set warning fire once for the window, then silence
    # it for the hundreds of single-instant refinement calls that follow.
    _, elevation, _ = look_angles_over_time(tle, site, grid)
    above = np.isfinite(elevation) & (elevation > mask_rad)

    passes: list[Pass] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StaleElementSetWarning)

        aos: datetime | None = start if above[0] else None
        starts_before = bool(above[0])

        for k in range(len(grid) - 1):
            if not above[k] and above[k + 1]:
                aos = _refine_crossing(
                    tle, site, grid[k], grid[k + 1], mask_rad, time_tolerance_s
                )
                starts_before = False
            elif above[k] and not above[k + 1] and aos is not None:
                los = _refine_crossing(
                    tle, site, grid[k + 1], grid[k], mask_rad, time_tolerance_s
                )
                passes.append(
                    _build_pass(
                        tle, site, aos, los, starts_before, False, time_tolerance_s
                    )
                )
                aos = None

        if above[-1] and aos is not None:
            passes.append(
                _build_pass(tle, site, aos, stop, starts_before, True, time_tolerance_s)
            )

    return passes


def _build_pass(
    tle: TLE,
    site: GroundSite,
    aos: datetime,
    los: datetime,
    starts_before_window: bool,
    ends_after_window: bool,
    tolerance_s: float,
) -> Pass:
    peak_time, peak_elevation = _refine_maximum(tle, site, aos, los, tolerance_s)
    return Pass(
        aos_utc=aos,
        los_utc=los,
        max_elevation_utc=peak_time,
        max_elevation_rad=peak_elevation,
        aos_azimuth_rad=_azimuth_rad(tle, site, aos),
        los_azimuth_rad=_azimuth_rad(tle, site, los),
        starts_before_window=starts_before_window,
        ends_after_window=ends_after_window,
    )
