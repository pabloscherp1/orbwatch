"""Delta-v and propellant budget with ESA assessment-study margins.

Each line of the budget carries a category that decides its margin, following
ESA's margin philosophy for science assessment studies (SRE-PA/2011.097,
Issue 1 Rev 3, 2012), which is consistent with ECSS-E-ST-10-02C:

- R-DV-11, 5%: accurately calculated manoeuvres, trajectory manoeuvres and
  detailed orbit maintenance.
- R-DV-12, 100%: general orbit maintenance not derived analytically.
- R-DV-13, 100%: attitude control and angular momentum management.
- R-DV-3: launcher dispersion must be assessed and included; its correction is
  calculated here, so it takes the 5% of R-DV-11.
- Allocations for work not yet designed are not analytically derived, so they
  take 100%, by the same reasoning as R-DV-12, until the design replaces them.

Mass follows the same document:

- R-M2-1: a system-level margin of at least 20% on the nominal dry mass.
- R-M1-5: propellant computed on the dry mass including that margin.
- R-M1-6: 2% of propellant added as residuals.

R-DV-2 asks for gravity losses of chemical engines to be added. Burns here are
impulsive and small, a few percent of orbital speed at most, where finite-burn
losses are well under a percent of the burn for any thruster that completes it
in minutes; they are not modelled.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

STANDARD_GRAVITY_M_S2: float = 9.80665

Category = Literal["calculated", "dispersion", "allocation", "maintenance", "attitude"]

MARGINS: dict[str, float] = {
    "calculated": 0.05,
    "dispersion": 0.05,
    "allocation": 1.00,
    "maintenance": 1.00,
    "attitude": 1.00,
}
"""Delta-v margin by category, as a fraction. See the module docstring."""

REQUIREMENTS: dict[str, str] = {
    "calculated": "R-DV-11",
    "dispersion": "R-DV-3, R-DV-11",
    "allocation": "R-DV-12 by analogy",
    "maintenance": "R-DV-12",
    "attitude": "R-DV-13",
}

SYSTEM_DRY_MASS_MARGIN: float = 0.20
PROPELLANT_RESIDUALS: float = 0.02

REFERENCE: str = (
    "ESA SRE-PA/2011.097, Margin philosophy for science assessment studies, "
    "Issue 1 Rev 3"
)


@dataclass(frozen=True)
class DeltaVItem:
    """One line of the budget.

    ``basis`` says where the number comes from, in a few words, so a reader
    can tell a computed manoeuvre from an allocation.
    """

    label: str
    dv_m_s: float
    category: Category
    basis: str

    @property
    def margin(self) -> float:
        return MARGINS[self.category]

    @property
    def dv_with_margin_m_s(self) -> float:
        return self.dv_m_s * (1.0 + self.margin)


def rocket_propellant_kg(dry_kg: float, dv_m_s: float, isp_s: float) -> float:
    """Propellant to give ``dv_m_s`` to a vehicle that ends at ``dry_kg``.

    Tsiolkovsky: m_prop = m_dry (exp(dv / (Isp g0)) - 1).
    """
    return float(dry_kg * np.expm1(dv_m_s / (isp_s * STANDARD_GRAVITY_M_S2)))


@dataclass(frozen=True)
class Budget:
    """Delta-v lines, their margins, and the propellant they need."""

    items: tuple[DeltaVItem, ...]
    nominal_dry_mass_kg: float
    isp_s: float

    @property
    def dv_nominal_m_s(self) -> float:
        return float(sum(i.dv_m_s for i in self.items))

    @property
    def dv_with_margins_m_s(self) -> float:
        return float(sum(i.dv_with_margin_m_s for i in self.items))

    @property
    def dry_mass_kg(self) -> float:
        """Dry mass including the system-level margin (R-M2-1)."""
        return self.nominal_dry_mass_kg * (1.0 + SYSTEM_DRY_MASS_MARGIN)

    @property
    def propellant_kg(self) -> float:
        """Propellant for the margined delta-v on the margined dry mass, plus
        residuals (R-M1-5, R-M1-6)."""
        usable = rocket_propellant_kg(
            self.dry_mass_kg, self.dv_with_margins_m_s, self.isp_s
        )
        return usable * (1.0 + PROPELLANT_RESIDUALS)

    @property
    def wet_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_kg


def build_budget(
    items: Sequence[DeltaVItem], nominal_dry_mass_kg: float, isp_s: float
) -> Budget:
    if nominal_dry_mass_kg <= 0.0 or isp_s <= 0.0:
        raise ValueError("dry mass and specific impulse must be positive")
    return Budget(tuple(items), nominal_dry_mass_kg, isp_s)
