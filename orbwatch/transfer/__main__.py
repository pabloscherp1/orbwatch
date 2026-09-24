"""Command line: plan an inspection rendezvous with a catalogued object.

    python -m orbwatch.transfer 27386
    python -m orbwatch.transfer 27386 --dropoff-altitude 550 --ltan-offset -2 \\
        --max-days 120 --dry-mass 150 --isp 220

The target's current element set comes from Celestrak. The drop-off is a
sun-synchronous rideshare orbit whose local time of ascending node is the
target's plus an offset in hours.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

import numpy as np

from orbwatch.budget.budget import REFERENCE, REQUIREMENTS, Budget, build_budget
from orbwatch.catalog.sources import CatalogFetchError, fetch_celestrak_tle
from orbwatch.transfer.mission import (
    MissionAssumptions,
    RendezvousPlan,
    budget_items,
    local_time_of_ascending_node_h,
    orbit_from_tle,
    plan_rendezvous,
    sun_synchronous_orbit,
)

SHOWN_DAYS: float = 730.0
"""Longest wait worth printing in the trade; beyond it the drift orbit is so
close to the target's that the planes barely move relative to each other."""


def _clock(hours: float) -> str:
    minutes = round(hours * 60.0) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def format_plan(plan: RendezvousPlan, name: str) -> str:
    d, t = plan.dropoff, plan.target
    epoch = d.epoch_utc
    c = plan.chosen
    lines = [
        f"RENDEZVOUS WITH {name}",
        f"  target     {t.altitude_km:6.1f} km,"
        f" i {np.rad2deg(t.inclination_rad):.3f} deg,"
        f" ascending node at {_clock(local_time_of_ascending_node_h(t, epoch))}"
        " local time",
        f"  drop-off   {d.altitude_km:6.1f} km sun-synchronous,"
        f" i {np.rad2deg(d.inclination_rad):.3f} deg,"
        f" ascending node at {_clock(local_time_of_ascending_node_h(d, epoch))}",
        f"  gap        {np.rad2deg(plan.delta_raan_rad):+.2f} deg of node,"
        f" {np.rad2deg(plan.delta_inclination_rad):+.3f} deg of inclination,"
        f" on {epoch:%Y-%m-%d}",
        f"  direct     turning the plane by thrust costs"
        f" {plan.direct.dv_km_s * 1000:.0f} m/s",
        "",
        "TIME AGAINST DELTA-V, drifting on J2",
        "     days     m/s   drift orbit",
    ]
    shown = [o for o in plan.front if o.total_days <= SHOWN_DAYS]
    step = max(1, len(shown) // 12)
    for o in shown[::step]:
        lines.append(
            f"  {o.total_days:7.1f} {o.dv_km_s * 1000:7.1f}   {o.altitude_km:4.0f} km,"
            f" i {np.rad2deg(o.inclination_rad - d.inclination_rad):+.2f} deg"
        )
    a = plan.approach
    lines += [
        "",
        f"CHOSEN, cheapest within {plan.max_days:g} days",
        f"  1  enter drift orbit {c.altitude_km:.0f} km,"
        f" i {np.rad2deg(c.inclination_rad):.3f} deg"
        f"{c.enter.dv_km_s * 1000:14.1f} m/s",
        f"  2  wait {c.wait_days:.1f} days for J2 to line the planes up",
        f"  3  time the exit within one synodic period, up to"
        f" {c.phasing_days:.1f} days",
        f"  4  transfer into the target's orbit{c.leave.dv_km_s * 1000:22.1f} m/s",
        f"  5  Lambert approach, {a.far_km:g} to {a.hold_km:g} km behind,"
        f" {a.time_of_flight_s / 60:.0f} min{a.dv_km_s * 1000:9.1f} m/s",
        f"  at the hold point on {plan.arrival_utc:%Y-%m-%d}",
    ]
    return "\n".join(lines)


def format_budget(budget: Budget, assumptions: MissionAssumptions) -> str:
    lines = [
        "",
        f"DELTA-V BUDGET, margins per {REFERENCE}",
        f"  {'line':<42}{'m/s':>7}{'margin':>8}{'with':>8}  basis",
    ]
    for item in budget.items:
        lines.append(
            f"  {item.label:<42}{item.dv_m_s:7.1f}{100 * item.margin:7.0f}%"
            f"{item.dv_with_margin_m_s:8.1f}  {item.basis}"
            f" ({REQUIREMENTS[item.category]})"
        )
    lines += [
        f"  {'total':<42}{budget.dv_nominal_m_s:7.1f}{'':8}"
        f"{budget.dv_with_margins_m_s:8.1f}",
        "",
        "MASS",
        f"  dry {budget.nominal_dry_mass_kg:.0f} kg nominal,"
        f" {budget.dry_mass_kg:.0f} kg with the 20% system margin (R-M2-1);"
        f" Isp {budget.isp_s:.0f} s",
        f"  propellant {budget.propellant_kg:.1f} kg including 2% residuals"
        f" (R-M1-6); wet {budget.wet_mass_kg:.1f} kg",
        "",
        "ASSUMPTIONS",
        f"  injection error {assumptions.injection_altitude_error_km:g} km and"
        f" {assumptions.injection_inclination_error_deg:g} deg;"
        f" proximity allocation {assumptions.proximity_allocation_m_s:g} m/s;"
        f" {assumptions.operations_days:g} days of operations;"
        f" disposal perigee {assumptions.disposal_perigee_km:g} km",
        "  mean J2 dynamics, circular orbits, impulsive burns, no drag",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m orbwatch.transfer",
        description="Plan a rendezvous from a rideshare drop-off to a catalogued"
        " object.",
    )
    parser.add_argument("norad", type=int, help="target NORAD catalogue number")
    parser.add_argument("--dropoff-altitude", type=float, default=525.0)
    parser.add_argument(
        "--ltan-offset",
        type=float,
        default=-1.0,
        help="drop-off local time of ascending node minus the target's, hours",
    )
    parser.add_argument("--max-days", type=float, default=180.0)
    parser.add_argument("--dry-mass", type=float, default=150.0, help="nominal, kg")
    parser.add_argument("--isp", type=float, default=220.0, help="seconds")
    parser.add_argument("--epoch", help="drop-off date, YYYY-MM-DD (default today)")
    args = parser.parse_args(argv)

    try:
        tle = fetch_celestrak_tle(args.norad)
    except CatalogFetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    epoch = (
        datetime.fromisoformat(args.epoch).replace(tzinfo=UTC)
        if args.epoch
        else datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    )
    target = orbit_from_tle(tle)
    ltan = local_time_of_ascending_node_h(target, epoch) + args.ltan_offset
    dropoff = sun_synchronous_orbit(args.dropoff_altitude, ltan % 24.0, epoch)
    try:
        plan = plan_rendezvous(dropoff, target, max_days=args.max_days)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    assumptions = MissionAssumptions()
    budget = build_budget(budget_items(plan, assumptions), args.dry_mass, args.isp)
    name = f"{tle.name or 'NORAD'} (NORAD {args.norad})"
    print(format_plan(plan, name))
    print(format_budget(budget, assumptions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
