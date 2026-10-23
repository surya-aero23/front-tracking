from fractions import Fraction

import pytest

from fronttrack import (
    BuckleyLeverett,
    Burgers,
    FrontTracker,
    PiecewiseConstant,
    exact_box_solution,
    l1_error,
    riemann_data,
    run_statistics,
)

BOX = PiecewiseConstant((0, 1), (0, 1, 0))


def test_statistics_of_the_box():
    # delta = 1/4: four fan fronts and one shock.  The shock absorbs the fan
    # one front at a time, but the last pair, 0 -> 1/4 and 1/4 -> 0, moves at
    # one common speed and never meets: three interactions.
    solution = FrontTracker(Burgers(), Fraction(1, 4)).solve(BOX, 20)
    stats = run_statistics(solution)
    assert stats.interactions == 3
    assert stats.fronts == 5 + 3
    assert stats.max_active == 5
    assert stats.denominator_digits <= 2


def test_burgers_never_has_more_fronts_than_at_the_start():
    solution = FrontTracker(Burgers(), Fraction(1, 16)).solve(BOX, 20)
    assert run_statistics(solution).max_active == len(solution.active_fronts(0))


@pytest.mark.parametrize("flux", [Burgers(), BuckleyLeverett()])
def test_work_grows_linearly_in_one_over_delta(flux):
    counts = [
        run_statistics(FrontTracker(flux, Fraction(1, n)).solve(BOX, 20)).interactions
        for n in (16, 32, 64)
    ]
    ratios = [b / a for a, b in zip(counts, counts[1:])]
    assert all(1.6 < ratio < 2.4 for ratio in ratios)


def test_burgers_denominators_stay_short():
    # Speeds are multiples of delta/2, so collision times and places keep
    # small denominators; nonconvex fluxes have no such structure.
    burgers = FrontTracker(Burgers(), Fraction(1, 64)).solve(BOX, 20)
    assert run_statistics(burgers).denominator_digits <= 4


@pytest.mark.parametrize("delta", [Fraction(1, 8), Fraction(1, 32)])
def test_box_error_is_within_the_flux_perturbation_bound(delta):
    # ||u_delta - u|| <= t * (delta / 2) * TV(u0), and TV(u0) = 2 for the box.
    t = 3
    solution = FrontTracker(Burgers(), delta).solve(BOX, t)
    error = l1_error(solution, exact_box_solution(), t, (-1, 4),
                     extra_breaks=[0, 6 ** 0.5])
    assert error <= float(t * delta / 2 * 2)


def test_riemann_rarefaction_error_is_half_the_bound():
    # The staircase misses the fan by a triangle of area delta t / 4 per unit
    # jump, against the bound delta t / 2.
    delta = Fraction(1, 16)
    solution = FrontTracker(Burgers(), delta).solve(riemann_data(0, 1), 1)
    error = l1_error(solution, lambda x, t: min(max(x / t, 0), 1), 1, (-1, 2),
                     extra_breaks=[0, 1])
    assert error == pytest.approx(float(delta / 4), rel=1e-3)
