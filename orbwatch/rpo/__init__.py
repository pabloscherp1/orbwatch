"""Proximity operations: Clohessy-Wiltshire relative motion, curvilinear
frames, approach and inspection design, passive safety and Monte Carlo."""

from dataclasses import dataclass

import numpy as np

from orbwatch.catalog.elements import MU_EARTH_KM3_S2
from orbwatch.rpo.approach import (
    ProximityDesign,
    SafetyCheck,
    design_inspection,
    missed_burn_checks,
)
from orbwatch.rpo.monte_carlo import Dispersions, MonteCarloResult, run_monte_carlo

PROXIMITY_PERCENTILE: float = 99.0
"""The Monte Carlo percentile carried into the delta-v budget."""


@dataclass(frozen=True)
class ProximityPlan:
    design: ProximityDesign
    checks: tuple[SafetyCheck, ...]
    monte_carlo: MonteCarloResult

    @property
    def budget_dv_m_s(self) -> float:
        return self.monte_carlo.dv_percentile(PROXIMITY_PERCENTILE)


def plan_proximity(
    target_a_km: float,
    dispersions: Dispersions | None = None,
    runs: int = 1000,
    seed: int = 1,
    **design: float,
) -> ProximityPlan:
    """Design, safety-check and Monte Carlo the operations around a target.

    ``design`` passes through to :func:`design_inspection`.
    """
    n = float(np.sqrt(MU_EARTH_KM3_S2 / target_a_km**3))
    nominal = design_inspection(n, **design)
    return ProximityPlan(
        design=nominal,
        checks=tuple(missed_burn_checks(nominal)),
        monte_carlo=run_monte_carlo(nominal, dispersions, runs=runs, seed=seed),
    )


__all__ = [
    "PROXIMITY_PERCENTILE",
    "Dispersions",
    "MonteCarloResult",
    "ProximityDesign",
    "ProximityPlan",
    "SafetyCheck",
    "design_inspection",
    "missed_burn_checks",
    "plan_proximity",
    "run_monte_carlo",
]
