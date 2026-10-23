from fractions import Fraction

import numpy as np
import pytest

from fronttrack import Burgers, FrontTracker, Periodic, l1_error, project
from fronttrack.exact import exact_sine_solution


def sine(x):
    return np.sin(2 * np.pi * x)


def test_constant_function_projects_to_constant_data():
    data = project(lambda x: 0.5 + 0 * x, (0, 1), 8, Fraction(1, 4))
    assert data.breakpoints == () and data.values == (Fraction(1, 2),)


def test_values_are_rounded_to_the_delta_grid():
    delta = Fraction(1, 8)
    data = project(sine, (0, 1), 16, delta)
    assert all((value / delta).denominator == 1 for value in data.values)
    assert data.u_max == 1 and data.u_min == -1


def test_both_rules_are_exact_for_a_ramp():
    # For u0 = x the cell averages over [0, 1/2] and [1/2, 1] are the
    # midpoint values 1/4 and 3/4, already on the delta grid.
    ramp = lambda x: x  # noqa: E731
    average = project(ramp, (0, 1), 2, Fraction(1, 4))
    assert average.values == (Fraction(1, 4), Fraction(3, 4))
    midpoint = project(ramp, (0, 1), 2, Fraction(1, 4), rule="midpoint")
    assert midpoint.values == average.values


def test_projected_data_fits_a_periodic_domain():
    data = project(sine, (0, 1), 16, Fraction(1, 8))
    assert all(0 < x < 1 for x in data.breakpoints)
    FrontTracker(Burgers(), Fraction(1, 8), Periodic(0, 1)).solve(data, Fraction(1, 10))


def test_projection_rejects_bad_parameters():
    with pytest.raises(ValueError):
        project(sine, (0, 1), 0, Fraction(1, 8))
    with pytest.raises(ValueError):
        project(sine, (1, 0), 4, Fraction(1, 8))
    with pytest.raises(ValueError):
        project(sine, (0, 1), 4, 0)
    with pytest.raises(ValueError):
        project(sine, (0, 1), 4, Fraction(1, 8), rule="trapezoid")


def test_exact_sine_solution_starts_from_the_data_and_is_odd():
    solution = exact_sine_solution()
    for x in (0.1, 0.3, 0.45):
        assert solution(x, 0) == pytest.approx(np.sin(2 * np.pi * x), abs=1e-12)
        for t in (0.1, 0.3, 1.0):
            assert solution(1 - x, t) == pytest.approx(-solution(x, t), abs=1e-12)
    assert solution.breaking_time == pytest.approx(1 / (2 * np.pi))


def test_exact_sine_solution_has_a_shock_only_after_breaking():
    solution = exact_sine_solution()
    before = solution(0.5 - 1e-9, 0.1) - solution(0.5 + 1e-9, 0.1)
    after = solution(0.5 - 1e-9, 0.3) - solution(0.5 + 1e-9, 0.3)
    assert before == pytest.approx(0, abs=1e-6)
    assert after > 1.5


def test_front_tracking_converges_on_sine_data_after_the_shock_forms():
    reference = exact_sine_solution()
    errors = []
    for cells in (32, 64):
        delta = Fraction(2, cells)
        data = project(sine, (0, 1), cells, delta)
        solution = FrontTracker(Burgers(), delta, Periodic(0, 1)).solve(data, Fraction(3, 10))
        errors.append(l1_error(solution, reference, Fraction(3, 10), (0, 1),
                               extra_breaks=[0.5]))
    assert errors[1] < errors[0] / 1.7


def test_first_collision_approximates_the_breaking_time():
    cells, delta = 128, Fraction(1, 128 * 16)
    data = project(sine, (0, 1), cells, delta)
    solution = FrontTracker(Burgers(), delta, Periodic(0, 1)).solve(data, Fraction(1, 5))
    first = float(solution.interactions[0].time)
    assert first == pytest.approx(1 / (2 * np.pi), abs=2e-3)
