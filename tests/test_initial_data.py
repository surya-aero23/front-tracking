from fractions import Fraction

import pytest

from fronttrack.initial_data import PiecewiseConstant, riemann_data


def test_riemann_data_is_a_single_jump():
    data = riemann_data(1, 0)
    assert data.breakpoints == (Fraction(0),)
    assert data.values == (Fraction(1), Fraction(0))
    assert data(-1) == 1 and data(1) == 0


def test_a_breakpoint_takes_the_right_hand_value():
    data = riemann_data(1, 0)
    assert data(0) == 0


def test_value_count_must_match_breakpoints():
    with pytest.raises(ValueError):
        PiecewiseConstant((0, 1), (1, 0))


def test_breakpoints_must_increase():
    with pytest.raises(ValueError):
        PiecewiseConstant((1, 0), (1, 0, 1))


def test_total_variation_sums_the_jumps():
    assert PiecewiseConstant((0, 1), (0, 1, 0)).total_variation() == 2


def test_mass_integrates_over_a_window():
    data = PiecewiseConstant((0, 1), (0, 1, 0))
    assert data.mass(-5, 5) == 1
    assert data.mass(Fraction(1, 2), 5) == Fraction(1, 2)


def test_mass_window_may_sit_inside_one_cell():
    data = riemann_data(2, 0)
    assert data.mass(-2, -1) == 2


def test_from_jumps_drops_redundant_breakpoints():
    data = PiecewiseConstant.from_jumps(0, [(0, 1), (1, 1), (2, 0)])
    assert data.breakpoints == (Fraction(0), Fraction(2))
    assert data.values == (Fraction(0), Fraction(1), Fraction(0))


def test_from_jumps_collapses_a_common_position():
    # Several fronts at one point, as at the instant of an interaction.
    data = PiecewiseConstant.from_jumps(0, [(1, 2), (1, 3)])
    assert data.breakpoints == (Fraction(1),)
    assert data.values == (Fraction(0), Fraction(3))


def test_from_jumps_can_collapse_to_a_constant():
    data = PiecewiseConstant.from_jumps(1, [(0, 2), (0, 1)])
    assert data.breakpoints == ()
    assert data.values == (Fraction(1),)
