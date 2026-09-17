"""Tests for conversion between inertial Cartesian state and classical elements.

Two kinds of test here, and they catch different things.

**Hand-derived cases.** Every expected number is derivable from the two-body
problem with no textbook lookup, no reference implementation and no fitted
tolerance. Each case states its derivation so a reader can check the test is
itself correct, which matters more than usual because a wrong test that passes
is worse than no test. These pin the absolute answer, but only at special
geometries where the algebra stays tractable by hand.

**Round-trip cases.** State to elements and back, at arbitrary angles where no
hand derivation is practical. These cannot catch an error shared by both
directions, such as a wrong sign convention applied consistently, so they are
not a substitute for the hand cases. What they do catch is quadrant errors,
basis handedness errors and the degenerate-branch conventions failing to invert.

Angular tolerance is 1e-9 rad rather than machine epsilon: loose enough to
survive an ``arccos`` somewhere in the chain, tight enough at about 0.2
milliarcseconds to catch any real error.
"""

import numpy as np
import pytest

from orbwatch.catalog.elements import (
    MU_EARTH_KM3_S2,
    OrbitalElements,
    elements_to_state,
    state_to_elements,
)

TWO_PI = 2.0 * np.pi
ANGLE_TOL_RAD = 1e-9


def wrapped_difference_rad(angle_rad: float, target_rad: float) -> float:
    """Smallest absolute difference between two angles, accounting for wrap."""
    difference = (angle_rad - target_rad) % TWO_PI
    return min(difference, TWO_PI - difference)


def assert_angle(angle_rad: float, target_rad: float, name: str) -> None:
    difference = wrapped_difference_rad(angle_rad, target_rad)
    assert difference < ANGLE_TOL_RAD, (
        f"{name}: got {angle_rad:.12f} rad, expected {target_rad:.12f} rad, "
        f"off by {difference:.3e} rad"
    )


# --------------------------------------------------------------------------
# Hand-derived cases
# --------------------------------------------------------------------------


def test_circular_inclined_orbit_at_ascending_node() -> None:
    """Circular, 45 deg inclined, spacecraft exactly on the ascending node.

    Derivation. With r = a x_hat and v the circular speed along
    (0, cos 45, sin 45):

    - Speed is circular so v^2 - mu/r = 0, and r is perpendicular to v so
      r . v = 0. Both terms of the eccentricity vector vanish, giving e = 0.
    - h = r x v = a v_c (0, -sin 45, cos 45), so cos i = h_z / h = cos 45.
    - n = z_hat x h points along +x_hat, so RAAN = 0.
    - The spacecraft sits on the node, so the argument of latitude is 0.

    The orbit is circular, so the argument of periapsis is undefined and the
    documented convention puts the whole in-plane angle into the true anomaly.
    This asserts the argument of latitude, so it holds under either convention.
    """
    a_km = 7000.0
    v_circular_km_s = np.sqrt(MU_EARTH_KM3_S2 / a_km)
    root_half = np.sqrt(0.5)

    r_eci_km = np.array([a_km, 0.0, 0.0])
    v_eci_km_s = v_circular_km_s * np.array([0.0, root_half, root_half])

    el = state_to_elements(r_eci_km, v_eci_km_s)

    assert el.a_km == pytest.approx(a_km, rel=1e-12)
    assert el.e == pytest.approx(0.0, abs=1e-12)
    assert_angle(el.i_rad, np.pi / 4.0, "inclination")
    assert_angle(el.raan_rad, 0.0, "raan")
    assert_angle(el.argument_of_latitude_rad, 0.0, "argument of latitude")


def test_circular_equatorial_orbit_hits_both_singularities() -> None:
    """Geostationary-radius circular equatorial orbit.

    This is the case that matters for the GEO case study, because it is
    near-circular and near-equatorial at once, so both degenerate branches fire
    together. Periapsis does not exist and neither does the line of nodes.

    Derivation. r = a x_hat and v = v_c y_hat give h = a v_c z_hat, so i = 0
    exactly and the node vector is identically zero. Under the documented
    convention RAAN = 0, in-plane angles are measured from x_hat, and with the
    spacecraft on x_hat the argument of latitude is 0.
    """
    a_km = 42164.0
    v_circular_km_s = np.sqrt(MU_EARTH_KM3_S2 / a_km)

    r_eci_km = np.array([a_km, 0.0, 0.0])
    v_eci_km_s = v_circular_km_s * np.array([0.0, 1.0, 0.0])

    el = state_to_elements(r_eci_km, v_eci_km_s)

    assert el.a_km == pytest.approx(a_km, rel=1e-12)
    assert el.e == pytest.approx(0.0, abs=1e-12)
    assert_angle(el.i_rad, 0.0, "inclination")
    assert_angle(el.raan_rad, 0.0, "raan")
    assert_angle(el.argument_of_latitude_rad, 0.0, "argument of latitude")


def test_elliptical_inclined_orbit_at_periapsis() -> None:
    """Eccentric, 30 deg inclined, spacecraft at periapsis on the node.

    Derivation. Choose a = 10000 km, e = 0.2, i = 30 deg, RAAN = 0, argp = 0.
    Periapsis then lies on the ascending node along x_hat at radius
    r_p = a(1 - e) = 8000 km, and periapsis velocity is perpendicular to r,
    in-plane and along the direction of motion, which for this geometry is
    (0, cos 30, sin 30), with magnitude

        v_p = sqrt( mu (1 + e) / (a (1 - e)) ).

    Then v_p^2 - mu/r_p = mu e / r_p and r . v = 0, so the eccentricity vector
    is exactly 0.2 x_hat. It points at the node, so argp = 0, and the
    spacecraft is at periapsis, so the true anomaly is 0.
    """
    a_km = 10000.0
    e_expected = 0.2
    i_expected_rad = np.deg2rad(30.0)

    r_periapsis_km = a_km * (1.0 - e_expected)
    v_periapsis_km_s = np.sqrt(
        MU_EARTH_KM3_S2 * (1.0 + e_expected) / (a_km * (1.0 - e_expected))
    )

    r_eci_km = np.array([r_periapsis_km, 0.0, 0.0])
    v_eci_km_s = v_periapsis_km_s * np.array(
        [0.0, np.cos(i_expected_rad), np.sin(i_expected_rad)]
    )

    el = state_to_elements(r_eci_km, v_eci_km_s)

    assert el.a_km == pytest.approx(a_km, rel=1e-12)
    assert el.e == pytest.approx(e_expected, rel=1e-12)
    assert_angle(el.i_rad, i_expected_rad, "inclination")
    assert_angle(el.raan_rad, 0.0, "raan")
    assert_angle(el.argp_rad, 0.0, "argument of periapsis")
    assert_angle(el.nu_rad, 0.0, "true anomaly")


def test_elements_to_state_reproduces_the_hand_derived_periapsis_state() -> None:
    """The inverse direction, pinned against the same hand-derived geometry."""
    a_km = 10000.0
    e = 0.2
    i_rad = np.deg2rad(30.0)

    el = OrbitalElements(
        a_km=a_km, e=e, i_rad=i_rad, raan_rad=0.0, argp_rad=0.0, nu_rad=0.0
    )
    r_eci_km, v_eci_km_s = elements_to_state(el)

    v_periapsis_km_s = np.sqrt(MU_EARTH_KM3_S2 * (1.0 + e) / (a_km * (1.0 - e)))
    expected_r_km = np.array([a_km * (1.0 - e), 0.0, 0.0])
    expected_v_km_s = v_periapsis_km_s * np.array([0.0, np.cos(i_rad), np.sin(i_rad)])

    assert r_eci_km == pytest.approx(expected_r_km, rel=1e-12, abs=1e-9)
    assert v_eci_km_s == pytest.approx(expected_v_km_s, rel=1e-12, abs=1e-12)


# --------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------

ROUND_TRIP_CASES = {
    # Low Earth orbit at the ISS inclination, mildly eccentric, past apoapsis.
    "leo_iss_like": (7000.0, 0.001, 51.6, 120.0, 30.0, 200.0),
    # Near-geostationary. Small but above-threshold eccentricity and
    # inclination, so both degenerate branches stay switched off and the
    # non-degenerate path is exercised right next to its singularities.
    "near_geo": (42164.0, 2.0e-4, 0.05, 80.0, 270.0, 45.0),
    # Highly eccentric at the critical inclination, Molniya-like.
    "molniya_like": (26560.0, 0.72, 63.4, 200.0, 270.0, 10.0),
    # Retrograde, to catch a handedness error that a prograde-only set misses.
    "retrograde": (8000.0, 0.05, 120.0, 45.0, 135.0, 300.0),
    # Sun-synchronous-ish, near-polar.
    "near_polar": (7200.0, 0.0015, 98.2, 310.0, 90.0, 170.0),
}


@pytest.mark.parametrize(
    ("a_km", "e", "i_deg", "raan_deg", "argp_deg", "nu_deg"),
    ROUND_TRIP_CASES.values(),
    ids=list(ROUND_TRIP_CASES),
)
def test_round_trip_closes_for_non_degenerate_orbits(
    a_km: float,
    e: float,
    i_deg: float,
    raan_deg: float,
    argp_deg: float,
    nu_deg: float,
) -> None:
    """Elements to state and back must return the same elements."""
    original = OrbitalElements(
        a_km=a_km,
        e=e,
        i_rad=np.deg2rad(i_deg),
        raan_rad=np.deg2rad(raan_deg),
        argp_rad=np.deg2rad(argp_deg),
        nu_rad=np.deg2rad(nu_deg),
    )

    r_eci_km, v_eci_km_s = elements_to_state(original)
    recovered = state_to_elements(r_eci_km, v_eci_km_s)

    assert recovered.a_km == pytest.approx(original.a_km, rel=1e-11)
    assert recovered.e == pytest.approx(original.e, rel=1e-9, abs=1e-12)
    assert_angle(recovered.i_rad, original.i_rad, "inclination")
    assert_angle(recovered.raan_rad, original.raan_rad, "raan")
    assert_angle(recovered.argp_rad, original.argp_rad, "argument of periapsis")
    assert_angle(recovered.nu_rad, original.nu_rad, "true anomaly")


def test_round_trip_closes_for_a_circular_inclined_orbit() -> None:
    """The circular convention must invert exactly.

    With e = 0 the argument of periapsis is undefined, so the round trip is
    only required to preserve the argument of latitude, which is the physically
    meaningful angle.
    """
    original = OrbitalElements(
        a_km=7000.0,
        e=0.0,
        i_rad=np.deg2rad(45.0),
        raan_rad=np.deg2rad(60.0),
        argp_rad=0.0,
        nu_rad=np.deg2rad(215.0),
    )

    r_eci_km, v_eci_km_s = elements_to_state(original)
    recovered = state_to_elements(r_eci_km, v_eci_km_s)

    assert recovered.a_km == pytest.approx(original.a_km, rel=1e-11)
    assert recovered.e == pytest.approx(0.0, abs=1e-12)
    assert_angle(recovered.i_rad, original.i_rad, "inclination")
    assert_angle(recovered.raan_rad, original.raan_rad, "raan")
    assert_angle(
        recovered.argument_of_latitude_rad,
        original.argument_of_latitude_rad,
        "argument of latitude",
    )


def test_round_trip_closes_for_a_circular_equatorial_orbit() -> None:
    """Both degenerate branches at once, the GEO case.

    RAAN is set to zero in the input on purpose. With i = 0 the line of nodes
    does not exist, so the split of the total in-plane angle between RAAN, argp
    and true anomaly is arbitrary and only their sum survives a round trip. The
    documented convention returns RAAN = 0 and folds everything into the true
    anomaly, so seeding a non-zero RAAN here would be asserting a quantity the
    physics does not define.
    """
    original = OrbitalElements(
        a_km=42164.0,
        e=0.0,
        i_rad=0.0,
        raan_rad=0.0,
        argp_rad=0.0,
        nu_rad=np.deg2rad(137.0),
    )

    r_eci_km, v_eci_km_s = elements_to_state(original)
    recovered = state_to_elements(r_eci_km, v_eci_km_s)

    assert recovered.a_km == pytest.approx(original.a_km, rel=1e-11)
    assert recovered.e == pytest.approx(0.0, abs=1e-12)
    assert_angle(recovered.i_rad, 0.0, "inclination")
    assert_angle(recovered.raan_rad, 0.0, "raan")
    assert_angle(
        recovered.argument_of_latitude_rad,
        original.argument_of_latitude_rad,
        "argument of latitude",
    )


# --------------------------------------------------------------------------
# Derived quantities and guards
# --------------------------------------------------------------------------


def test_geostationary_period_is_one_sidereal_day() -> None:
    """A 42164.0 km semi-major axis should give a sidereal day to within a second.

    The sidereal day is 86164.1 s. This is a consistency check on the derived
    properties and on MU_EARTH_KM3_S2, not an independent measurement.
    """
    el = OrbitalElements(
        a_km=42164.0, e=0.0, i_rad=0.0, raan_rad=0.0, argp_rad=0.0, nu_rad=0.0
    )
    assert el.period_s == pytest.approx(86164.1, abs=1.0)


def test_apsides_bracket_the_semi_major_axis() -> None:
    el = OrbitalElements(
        a_km=26560.0,
        e=0.72,
        i_rad=np.deg2rad(63.4),
        raan_rad=0.0,
        argp_rad=0.0,
        nu_rad=0.0,
    )
    assert el.r_periapsis_km == pytest.approx(26560.0 * 0.28, rel=1e-12)
    assert el.r_apoapsis_km == pytest.approx(26560.0 * 1.72, rel=1e-12)
    assert el.r_periapsis_km < el.a_km < el.r_apoapsis_km


def test_near_parabolic_state_is_rejected() -> None:
    """Escape speed gives zero specific energy, where the semi-major axis
    diverges. The function must refuse rather than return a huge number."""
    r_km = 7000.0
    v_escape_km_s = np.sqrt(2.0 * MU_EARTH_KM3_S2 / r_km)

    r_eci_km = np.array([r_km, 0.0, 0.0])
    v_eci_km_s = v_escape_km_s * np.array([0.0, 1.0, 0.0])

    with pytest.raises(ValueError, match="parabolic"):
        state_to_elements(r_eci_km, v_eci_km_s)


def test_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        state_to_elements(np.array([7000.0, 0.0]), np.array([0.0, 7.5, 0.0]))


def test_unbound_elements_are_rejected() -> None:
    el = OrbitalElements(
        a_km=10000.0, e=1.4, i_rad=0.0, raan_rad=0.0, argp_rad=0.0, nu_rad=0.0
    )
    with pytest.raises(ValueError, match="bound ellipse"):
        elements_to_state(el)


def test_elements_are_immutable() -> None:
    """frozen=True means a downstream function cannot quietly mutate a shared
    element set. Attempting it is an error rather than a silent side effect."""
    el = OrbitalElements(
        a_km=7000.0, e=0.0, i_rad=0.0, raan_rad=0.0, argp_rad=0.0, nu_rad=0.0
    )
    with pytest.raises(AttributeError):
        el.a_km = 8000.0  # type: ignore[misc]
