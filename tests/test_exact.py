from fractions import Fraction

import pytest

from fronttrack.exact import exact_riemann_solution
from fronttrack.flux import BuckleyLeverett, Burgers, ConcaveBurgers


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


def test_buckley_leverett_shock_starts_at_the_welge_point():
    # For M = 1/2 the tangency f'(u) u = f(u) from u = 0 gives u = 1/sqrt(3).
    from fronttrack.exact import envelope_riemann_waves

    fan, shock = envelope_riemann_waves(1, 0, BuckleyLeverett())
    assert (fan.kind, shock.kind) == ("fan", "shock")
    assert fan.left == 1 and shock.right == 0
    assert shock.left == pytest.approx(3 ** -0.5, abs=1e-12)
    assert shock.speed == pytest.approx((1 + 3 ** 0.5) / 2, abs=1e-12)


def test_buckley_leverett_solution_is_monotone_with_a_jump_at_the_shock():
    solution = exact_riemann_solution(1, 0, BuckleyLeverett())
    speed = (1 + 3 ** 0.5) / 2
    assert solution(-0.1, 1) == 1
    assert solution(speed - 1e-9, 1) == pytest.approx(3 ** -0.5, abs=1e-6)
    assert solution(speed + 1e-9, 1) == 0
    xs = [0.1 * k for k in range(14)]
    values = [solution(x, 1) for x in xs]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_numerical_construction_reproduces_the_closed_forms():
    from fronttrack.exact import _numerical_riemann_solution

    for uL, uR in [(1, 0), (0, 1), (-1, 2), (2, -1)]:
        closed = exact_riemann_solution(uL, uR)
        numerical = _numerical_riemann_solution(uL, uR, Burgers())
        for x in [-1.5, -0.3, 0.2, 0.49, 0.51, 1.3, 2.5]:
            assert numerical(x, 1) == pytest.approx(float(closed(Fraction(str(x)), 1)),
                                                    abs=1e-9)


def test_front_tracking_converges_to_the_buckley_leverett_solution():
    from fronttrack import FrontTracker, l1_error, riemann_data

    reference = exact_riemann_solution(1, 0, BuckleyLeverett())
    errors = []
    for delta in (Fraction(1, 8), Fraction(1, 32)):
        solution = FrontTracker(BuckleyLeverett(), delta).solve(riemann_data(1, 0), 1)
        errors.append(l1_error(solution, reference, 1, (-0.5, 2),
                               extra_breaks=[0, (1 + 3 ** 0.5) / 2]))
    assert errors[1] < errors[0] / 3


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
