"""The solver is flux agnostic: Burgers is only the default.

These runs exercise a concave flux and the nonconvex Buckley-Leverett flux,
where the simple rule "decreasing jump gives a shock" fails and the envelope
construction of Lecture 8&9, section 5.13, does the work.
"""

from fractions import Fraction

import pytest

from fronttrack.diagnostics import (
    check_all_fronts,
    mass_defect,
    polynomial_bump,
    weak_form_residual,
)
from fronttrack.flux import BuckleyLeverett, ConcaveBurgers
from fronttrack.initial_data import PiecewiseConstant, riemann_data
from fronttrack.solver import FrontTracker

EIGHTH = Fraction(1, 8)


def test_concave_flux_spreads_a_decreasing_jump():
    # For f = -u^2/2 the roles are reversed, so u_l > u_r rarefies.
    solution = FrontTracker(ConcaveBurgers(), Fraction(1, 4)).solve(
        riemann_data(1, 0), 2
    )
    assert solution.n_fronts == 4
    assert all(front.speed < 0 for front in solution.fronts)
    assert all(report.ok for report in check_all_fronts(solution))


def test_concave_flux_keeps_an_increasing_jump_as_one_shock():
    solution = FrontTracker(ConcaveBurgers(), Fraction(1, 4)).solve(
        riemann_data(0, 1), 2
    )
    assert solution.n_fronts == 1
    assert solution.fronts[0].speed == Fraction(-1, 2)


def test_buckley_leverett_riemann_problem_is_a_composite_wave():
    # The chord from u = 1 to u = 0 crosses the graph, so the solution is a
    # rarefaction next to a shock rather than a single shock.
    solution = FrontTracker(BuckleyLeverett(), EIGHTH).solve(riemann_data(1, 0), 1)
    assert solution.n_fronts > 1
    jumps = [abs(front.jump) for front in solution.fronts]
    assert max(jumps) > min(jumps)
    assert all(report.ok for report in check_all_fronts(solution))


def test_buckley_leverett_speeds_increase_across_the_fan():
    solution = FrontTracker(BuckleyLeverett(), EIGHTH).solve(riemann_data(1, 0), 1)
    speeds = [front.speed for front in solution.fronts]
    assert all(a < b for a, b in zip(speeds, speeds[1:]))


def test_an_interaction_can_produce_several_waves():
    # With a nonconvex flux a collision need not reduce the number of fronts.
    solution = FrontTracker(BuckleyLeverett(), EIGHTH).solve(
        PiecewiseConstant((0, 1), (0, 1, 0)), 20
    )
    assert any(len(event.outgoing) > 1 for event in solution.interactions)
    assert all(report.ok for report in check_all_fronts(solution))


def test_three_fronts_can_annihilate():
    # The states 1/8 -> 1/4 -> 0 -> 1/8 have decreasing front speeds for the
    # Buckley-Leverett flux, so the three fronts can be placed to meet at one
    # point; the Riemann problem left behind is trivial and nothing comes out.
    data = PiecewiseConstant(
        (Fraction(-640, 561), Fraction(-8, 11), Fraction(-16, 51)),
        (EIGHTH, Fraction(1, 4), 0, EIGHTH),
    )
    solution = FrontTracker(BuckleyLeverett(), EIGHTH).solve(data, 2)
    assert solution.n_interactions == 1
    event = solution.interactions[0]
    assert (event.time, event.position) == (Fraction(1), Fraction(0))
    assert event.is_cancellation
    assert solution.state_at(2).values == (EIGHTH,)
    assert solution.total_variation(2) == 0


@pytest.mark.parametrize("flux", [ConcaveBurgers(), BuckleyLeverett()])
def test_conservation_and_the_weak_form_hold_for_any_flux(flux):
    data = PiecewiseConstant((0, 1), (0, Fraction(3, 4), Fraction(1, 4)))
    solution = FrontTracker(flux, EIGHTH).solve(data, 4)
    assert mass_defect(solution, 4, -40, 40) == 0
    bump = polynomial_bump(x_center=0.0, x_radius=20.0, t_center=0.0, t_radius=3.6)
    assert abs(weak_form_residual(solution, bump)) < 1e-10


def test_total_variation_does_not_increase_for_a_nonconvex_flux():
    data = PiecewiseConstant((-1, 0, 1), (Fraction(1, 4), 1, 0, Fraction(3, 4)))
    solution = FrontTracker(BuckleyLeverett(), EIGHTH).solve(data, 8)
    variations = [solution.total_variation(Fraction(k, 2)) for k in range(17)]
    assert all(b <= a for a, b in zip(variations, variations[1:]))
