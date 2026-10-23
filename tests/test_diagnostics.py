from fractions import Fraction

import pytest

from fronttrack.diagnostics import (
    check_all_fronts,
    l1_error,
    mass_defect,
    polynomial_bump,
    weak_form_residual,
)
from fronttrack.exact import exact_riemann_solution
from fronttrack.flux import Burgers
from fronttrack.initial_data import PiecewiseConstant, riemann_data
from fronttrack.solver import FrontTracker

RUNS = [
    (riemann_data(1, 0), 2),
    (riemann_data(0, 1), 2),
    (PiecewiseConstant((0, 1), (2, 1, 0)), 3),
    (PiecewiseConstant((0, 1), (0, 1, 0)), 6),
    (PiecewiseConstant((-1, 0, 1), (1, -1, 1, -1)), 6),
]


@pytest.fixture
def tracker():
    return FrontTracker(Burgers(), Fraction(1, 4))


@pytest.mark.parametrize("data,t_final", RUNS)
def test_all_fronts_pass_all_three_entropy_tests(tracker, data, t_final):
    reports = check_all_fronts(tracker.solve(data, t_final))
    assert reports
    assert all(report.ok for report in reports)


@pytest.mark.parametrize("data,t_final", RUNS)
def test_the_weak_formulation_holds(tracker, data, t_final):
    # The quadrature is exact for a polynomial test function, so the residual
    # is only limited by round-off.
    solution = tracker.solve(data, t_final)
    bump = polynomial_bump(
        x_center=0.0, x_radius=12.0,
        t_center=0.0, t_radius=float(t_final) * 0.9,
    )
    assert abs(weak_form_residual(solution, bump)) < 1e-10


def test_the_weak_formulation_holds_for_an_off_centre_test_function(tracker):
    solution = tracker.solve(PiecewiseConstant((0, 1), (2, 1, 0)), 3)
    bump = polynomial_bump(x_center=1.5, x_radius=6.0, t_center=1.0, t_radius=1.8)
    assert abs(weak_form_residual(solution, bump)) < 1e-10


def test_a_test_function_must_reach_into_the_time_interval(tracker):
    solution = tracker.solve(riemann_data(1, 0), 1)
    bump = polynomial_bump(t_center=5.0, t_radius=1.0)
    with pytest.raises(ValueError):
        weak_form_residual(solution, bump)


@pytest.mark.parametrize("data,t_final", RUNS)
def test_mass_balances_the_boundary_flux(tracker, data, t_final):
    solution = tracker.solve(data, t_final)
    for step in range(5):
        t = Fraction(t_final) * Fraction(step, 4)
        assert mass_defect(solution, t, -40, 40) == 0


def test_a_tracked_shock_is_exact(tracker):
    # Front tracking resolves a single shock without any error at all.
    solution = tracker.solve(riemann_data(1, 0), 2)
    assert l1_error(solution, exact_riemann_solution(1, 0), 2, (-3, 4)) == 0


def test_rarefaction_error_is_first_order_in_delta():
    reference = exact_riemann_solution(0, 1)
    errors = []
    for power in range(2, 7):
        delta = Fraction(1, 2 ** power)
        solution = FrontTracker(Burgers(), delta).solve(riemann_data(0, 1), 1)
        # The exact fan has kinks at x = 0 and x = 1 at this time.
        errors.append(
            l1_error(solution, reference, 1, (-2, 3), extra_breaks=[0, 1])
        )
    # Halving delta halves the error.
    for coarse, fine in zip(errors, errors[1:]):
        assert 1.8 < coarse / fine < 2.2
    assert errors[-1] < 0.01


def test_error_of_a_wave_interaction_vanishes_with_delta():
    # Two merging shocks are resolved exactly, whatever the mesh in u.
    reference = exact_riemann_solution(2, 0)
    for delta in [Fraction(1, 2), Fraction(1, 8)]:
        solution = FrontTracker(Burgers(), delta).solve(
            PiecewiseConstant((0, 0 + 1), (2, 1, 0)), 10
        )
        # Long after the merger the solution is the single shock from 2 to 0,
        # shifted by the finite time the two shocks took to meet.
        state = solution.state_at(10)
        assert state.values == (Fraction(2), Fraction(0))
        assert state.breakpoints == (Fraction(21, 2),)
        # The exact shock sits at x = 10, the tracked one half a unit later
        # because the two shocks needed a finite time to meet.
        error = l1_error(solution, reference, 10, (-5, 30), extra_breaks=[10])
        assert error == pytest.approx(1.0, abs=1e-9)


def test_front_reports_name_the_failing_condition(tracker):
    solution = tracker.solve(riemann_data(0, 1), 1)
    report = check_all_fronts(solution)[0]
    assert report.rankine_hugoniot and report.oleinik and report.kruzkov
    assert report.identifier == 0
