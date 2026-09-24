"""Tests for the delta-v and propellant budget and its ESA margins."""

import numpy as np
import pytest

from orbwatch.budget.budget import (
    MARGINS,
    STANDARD_GRAVITY_M_S2,
    DeltaVItem,
    build_budget,
    rocket_propellant_kg,
)


def test_margins_follow_the_esa_assessment_study_rules() -> None:
    assert MARGINS["calculated"] == 0.05, "R-DV-11"
    assert MARGINS["maintenance"] == 1.00, "R-DV-12"
    assert MARGINS["attitude"] == 1.00, "R-DV-13"
    assert DeltaVItem(
        "burn", 100.0, "calculated", ""
    ).dv_with_margin_m_s == pytest.approx(105.0)
    assert DeltaVItem("upkeep", 10.0, "maintenance", "").dv_with_margin_m_s == 20.0


def test_rocket_equation() -> None:
    isp = 220.0
    doubling = isp * STANDARD_GRAVITY_M_S2 * np.log(2.0)
    assert rocket_propellant_kg(100.0, doubling, isp) == pytest.approx(100.0)
    assert rocket_propellant_kg(100.0, 0.0, isp) == 0.0


def test_propellant_uses_margined_dry_mass_margined_dv_and_residuals() -> None:
    items = [
        DeltaVItem("transfer", 200.0, "calculated", ""),
        DeltaVItem("allocation", 10.0, "allocation", ""),
    ]
    budget = build_budget(items, nominal_dry_mass_kg=150.0, isp_s=220.0)
    assert budget.dv_nominal_m_s == pytest.approx(210.0)
    assert budget.dv_with_margins_m_s == pytest.approx(230.0)
    assert budget.dry_mass_kg == pytest.approx(180.0), "R-M2-1, 20%"
    expected = 1.02 * 180.0 * np.expm1(230.0 / (220.0 * STANDARD_GRAVITY_M_S2))
    assert budget.propellant_kg == pytest.approx(expected), "R-M1-5 and R-M1-6"
    assert budget.wet_mass_kg == pytest.approx(180.0 + expected)


def test_nonsense_inputs_are_refused() -> None:
    with pytest.raises(ValueError):
        build_budget([], 0.0, 220.0)
    with pytest.raises(ValueError):
        build_budget([], 100.0, -1.0)
