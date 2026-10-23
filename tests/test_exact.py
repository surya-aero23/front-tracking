from fractions import Fraction

import pytest

from fronttrack.exact import exact_riemann_solution
from fronttrack.flux import BuckleyLeverett, ConcaveBurgers


def test_shock_moves_at_the_average_of_the_states():
    solution = exact_riemann_solution(1, 0)
    # The shock sits at x = t/2.
    assert solution(Fraction(4, 10), 1) == 1
    assert solution(Fraction(6, 10), 1) == 0


def test_rarefaction_is_the_similarity_profile():
    solution = exact_riemann_solution(0, 1)
    assert solution(Fraction(-1, 2), 1) == 0
    assert solution(Fraction(1, 3), 1) == Fraction(1, 3)
    assert solution(2, 1) == 1


def test_rarefaction_scales_with_time():
    solution = exact_riemann_solution(0, 2)
    assert solution(1, 2) == Fraction(1, 2)


def test_concave_flux_swaps_shock_and_rarefaction():
    flux = ConcaveBurgers()
    # For f = -u^2/2 an increasing jump is the shock.
    increasing = exact_riemann_solution(0, 1, flux)
    assert increasing(Fraction(-6, 10), 1) == 0
    assert increasing(Fraction(-4, 10), 1) == 1
    decreasing = exact_riemann_solution(1, 0, flux)
    assert decreasing(Fraction(-1, 2), 1) == Fraction(1, 2)


def test_constant_data_is_constant():
    solution = exact_riemann_solution(1, 1)
    assert solution(5, 1) == 1


def test_time_must_be_positive():
    with pytest.raises(ValueError):
        exact_riemann_solution(1, 0)(0, 0)


def test_nonconvex_flux_has_no_closed_form_here():
    with pytest.raises(NotImplementedError):
        exact_riemann_solution(1, 0, BuckleyLeverett())


def test_box_solution_before_and_after_the_fan_catches_the_shock():
    from fronttrack.exact import exact_box_solution

    solution = exact_box_solution()
    # t = 1: fan on [0, 1], plateau up to the shock at 3/2.
    assert solution(Fraction(1, 2), 1) == Fraction(1, 2)
    assert solution(Fraction(5, 4), 1) == 1
    assert solution(Fraction(7, 4), 1) == 0
    # t = 8: the shock sits at sqrt(16) = 4.
    assert solution(Fraction(39, 10), 8) == Fraction(39, 80)
    assert solution(Fraction(41, 10), 8) == 0
