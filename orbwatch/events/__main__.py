"""Command line: analyse one object's element set history.

    python -m orbwatch.events 25544
    python -m orbwatch.events 40882 --days 365 --plot inmarsat.png

Needs Space-Track credentials in the environment or in ``.env`` in the working
directory. Downloads are cached under ``data/cache/spacetrack``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from orbwatch.catalog.sources import CatalogFetchError, fetch_celestrak_tle
from orbwatch.events.analysis import EventReport, analyse

ARCSEC_PER_RAD = 180.0 / np.pi * 3600.0


def _describe_change(kind: str, change: tuple[float, ...]) -> str:
    if kind == "in_plane":
        return f"{change[0]:+8.2f} km"
    return f"{np.linalg.norm(change) * ARCSEC_PER_RAD:7.1f} arcsec"


def format_report(report: EventReport, name: str) -> str:
    """Plain-text report of an analysis, for the terminal."""
    h = report.history
    q = h.quality
    regime = "geosynchronous" if h.regime == "geosynchronous" else "not geosynchronous"
    lines = [
        f"{name}  ·  NORAD {h.norad_id}  ·  {regime}",
        f"{h.epochs_utc[0]:%Y-%m-%d} to {h.epochs_utc[-1]:%Y-%m-%d}"
        f"  ({h.span_days:.0f} days)",
        "",
        "DATA QUALITY",
        f"  {q.records_in} element sets in  ·  {q.refits_merged} re-fits merged  ·  "
        f"{q.transients_rejected} transient sets rejected  ·  {len(h)} used",
        "",
        f"MANOEUVRES ({len(report.manoeuvres)})",
    ]
    if report.manoeuvres:
        lines.append(
            f"  {'window (UTC)':<33}{'axis':<14}{'profile':<11}"
            f"{'change':>14}{'delta-v':>11}{'sigma':>8}"
        )
        for m in report.manoeuvres:
            window = f"{m.start_utc:%Y-%m-%d %H:%M} to {m.end_utc:%m-%d %H:%M}"
            axis = "in-plane" if m.kind == "in_plane" else "out-of-plane"
            lines.append(
                f"  {window:<33}{axis:<14}{m.profile:<11}"
                f"{_describe_change(m.kind, m.change):>14}"
                f"{m.delta_v_m_s:8.2f} m/s{m.significance:8.0f}"
            )
    else:
        lines.append("  none detected")
    if report.drag_surges:
        lines += ["", f"DRAG SURGES ({len(report.drag_surges)}), natural, not burns"]
        for m in report.drag_surges:
            window = f"{m.start_utc:%Y-%m-%d %H:%M} to {m.end_utc:%m-%d %H:%M}"
            lines.append(
                f"  {window:<33}extra drop {m.change[0] * 1000:6.0f} m"
                f" over {m.window_hours / 24:.1f} days{m.significance:8.0f} sigma"
            )
    lines += [
        "",
        f"SET ASIDE  {len(report.below_floor)} below the delta-v floor  ·  "
        f"{len(report.incoherent)} incoherent  ·  {len(report.excursions)} in"
        " returned excursions (transients, not burns)",
        "",
        "BUDGET",
    ]
    for label, budget in (
        ("in-plane", report.in_plane_budget),
        ("out-of-plane", report.out_of_plane_budget),
    ):
        per_year = budget.per_year(budget.detected_m_s)
        line = (
            f"  {label:<14}detected {budget.detected_m_s:6.1f} m/s"
            f" ({per_year:.1f} per year)"
        )
        if budget.required_m_s is not None:
            closure = budget.closure
            line += f"  ·  required {budget.required_m_s:.1f} m/s"
            line += f"  ·  closure {100 * closure:.0f}%" if closure is not None else ""
        lines.append(line)
        lines.append(f"  {'':<14}{budget.method}")
    return "\n".join(lines)


def plot_report(report: EventReport, name: str, path: Path) -> Path:
    """Save a figure of the history with detected manoeuvres marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = report.history
    dates = np.array(h.epochs_utc)
    geo = h.regime == "geosynchronous"
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)

    axes[0].plot(dates, h.a_km, lw=0.9, color="#0b7a99")
    axes[0].set_ylabel("semi-major axis, km")
    if geo:
        deg = np.rad2deg(h.inclination_vector_rad)
        axes[1].plot(dates, deg[:, 0], lw=0.9, color="#0b7a99", label="i sin Ω")
        axes[1].plot(dates, deg[:, 1], lw=0.9, color="#8f5300", label="i cos Ω")
        axes[1].set_ylabel("inclination vector, degrees")
        axes[1].legend(loc="upper left", frameon=False)
    else:
        axes[1].plot(dates, np.rad2deg(h.inclination_rad), lw=0.9, color="#0b7a99")
        axes[1].set_ylabel("inclination, degrees")

    colours = {"impulsive": "#2f7a4d", "sustained": "#b3261e"}
    for m in report.manoeuvres:
        ax = axes[0] if m.kind == "in_plane" else axes[1]
        ax.axvspan(m.start_utc, m.end_utc, color=colours[m.profile], alpha=0.35, lw=0)
    for m in report.drag_surges:
        axes[0].axvspan(m.start_utc, m.end_utc, color="#c77700", alpha=0.35, lw=0)
    for when in h.quality.transient_epochs_utc:
        for ax in axes:
            ax.axvline(when, color="#b3261e", lw=0.6, alpha=0.25)

    ip, op = report.in_plane_budget, report.out_of_plane_budget
    parts = []
    for label, b in (("in-plane", ip), ("out-of-plane", op)):
        if b.closure is not None:
            parts.append(
                f"{label}: {b.detected_m_s:.1f} of {b.required_m_s:.1f} m/s required"
                f" ({100 * b.closure:.0f}%)"
            )
        elif b.detected_m_s > 0:
            parts.append(f"{label}: {b.detected_m_s:.1f} m/s detected")
    fig.suptitle(
        f"{name}  ·  NORAD {h.norad_id}  ·  {len(report.manoeuvres)} manoeuvres",
        x=0.01,
        ha="left",
    )
    axes[0].set_title(
        "   ".join(parts)
        + "\ngreen: impulsive manoeuvre   red: sustained   orange: drag surge"
        + "   faint red lines: rejected element sets",
        loc="left",
        fontsize=9,
        color="#56657a",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m orbwatch.events",
        description="Detect manoeuvres in one object's element set history.",
    )
    parser.add_argument("norad", type=int, help="NORAD catalogue number")
    parser.add_argument(
        "--days", type=int, default=365, help="history length, days (default 365)"
    )
    parser.add_argument(
        "--sigmas", type=float, default=5.0, help="detection threshold (default 5)"
    )
    parser.add_argument("--plot", type=Path, help="save a figure to this PNG file")
    args = parser.parse_args(argv)

    from orbwatch.catalog.spacetrack import SpaceTrackClient, SpaceTrackError

    try:
        client = SpaceTrackClient.from_env()
        stop = date.today()
        records = client.element_set_history(
            args.norad, stop - timedelta(days=args.days), stop
        )
    except SpaceTrackError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not records:
        print(
            f"no element sets for NORAD {args.norad} in the last {args.days} days",
            file=sys.stderr,
        )
        return 1

    try:
        name = fetch_celestrak_tle(args.norad).name or f"NORAD {args.norad}"
    except CatalogFetchError:
        name = f"NORAD {args.norad}"

    report = analyse(records, sigmas=args.sigmas)
    print(format_report(report, name))
    if args.plot:
        saved = plot_report(report, name, args.plot)
        print(f"\nfigure saved to {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
