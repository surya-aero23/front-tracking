from fractions import Fraction

import numpy as np
import pytest

from fronttrack import (
    Burgers,
    FrontTracker,
    GodunovSolver,
    RoeSolver,
    check_all_fronts,
    exact_riemann_solution,
    expansion_shock,
    l1_distance,
    polynomial_bump,
    riemann_data,
    roe_flux,
    weak_form_residual,
)


def test_expansion_shock_is_a_weak_solution():
    solution = expansion_shock(0, 1, t_final=2)
    reports = check_all_fronts(solution)
    assert len(reports) == 1
    assert reports[0].rankine_hugoniot
    bump = polynomial_bump(x_center=0.5, x_radius=2.0, t_center=0.5, t_radius=1.0)
    assert abs(weak_form_residual(solution, bump)) < 1e-12


def test_expansion_shock_fails_both_entropy_conditions():
    (report,) = check_all_fronts(expansion_shock(0, 1, t_final=2))
    assert not report.oleinik
    assert not report.kruzkov
    assert not report.ok


def test_two_weak_solutions_share_the_same_data():
    weak = expansion_shock(0, 1, t_final=2)
    entropy = FrontTracker(Burgers(), Fraction(1, 4)).solve(riemann_data(0, 1), 2)
    assert weak.state_at(0) == entropy.state_at(0)
    assert l1_distance(weak.state_at(2), entropy.state_at(2)) > 0


def test_a_decreasing_jump_is_already_the_entropy_shock():
    (report,) = check_all_fronts(expansion_shock(1, 0, t_final=1))
    assert report.ok


def test_roe_flux_upwinds_along_the_chord():
    left, right = np.array([-1.0, 2.0, 1.0]), np.array([1.0, 1.0, 2.0])
    assert roe_flux(Burgers(), left, right).tolist() == [0.5, 2.0, 0.5]


@pytest.mark.parametrize("cells", [100, 400])
def test_roe_keeps_a_transonic_expansion_shock(cells):
    reference = exact_riemann_solution(-1, 1)
    roe = RoeSolver(Burgers(), cells, (-2, 2)).solve(riemann_data(-1, 1), 1)
    # The jump from -1 to 1 never opens: the L1 distance to the fan stays 1.
    assert roe.l1_error(reference) == pytest.approx(1.0, abs=0.02)


def test_godunov_opens_the_same_rarefaction():
    reference = exact_riemann_solution(-1, 1)
    godunov = GodunovSolver(Burgers(), 400, (-2, 2)).solve(riemann_data(-1, 1), 1)
    assert godunov.l1_error(reference) < 0.05


def test_roe_is_fine_away_from_sonic_points():
    reference = exact_riemann_solution(1, 2)
    errors = [
        RoeSolver(Burgers(), cells, (-1, 4)).solve(riemann_data(1, 2), 1)
        .l1_error(reference)
        for cells in (100, 400)
    ]
    assert errors[1] < errors[0] / 2
