from fractions import Fraction

import numpy as np
import pytest

from fronttrack import (
    BuckleyLeverett,
    Burgers,
    ConcaveBurgers,
    FrontTracker,
    PiecewiseConstant,
    exact_riemann_solution,
    riemann_data,
)
from fronttrack.exact import exact_box_solution
from fronttrack.finite_volume import GodunovSolver, godunov_flux, lipschitz_bound


@pytest.mark.parametrize(
    "left, right, expected",
    [
        (1.0, 2.0, 0.5),    # both states move right: upwind from the left
        (-2.0, -1.0, 0.5),  # both move left: upwind from the right
        (-1.0, 1.0, 0.0),   # transonic rarefaction: f at the sonic point
        (1.0, -1.0, 0.5),   # stationary shock
        (2.0, -1.0, 2.0),   # shock moving right
    ],
)
def test_godunov_flux_for_burgers(left, right, expected):
    value = godunov_flux(Burgers(), np.array([left]), np.array([right]))
    assert value[0] == pytest.approx(expected)


def test_godunov_flux_for_concave_flux_swaps_min_and_max():
    # For f = -u^2/2 the sonic point is a maximum, met in a decreasing jump.
    value = godunov_flux(ConcaveBurgers(), np.array([1.0]), np.array([-1.0]))
    assert value[0] == pytest.approx(0.0)


def test_godunov_flux_is_consistent():
    u = np.linspace(-2, 2, 9)
    assert np.allclose(godunov_flux(Burgers(), u, u), 0.5 * u * u)


def test_float_flux_matches_exact_flux():
    u = np.linspace(0, 1, 11)
    for flux in (Burgers(), ConcaveBurgers(), BuckleyLeverett()):
        exact = np.array([float(flux.f(value)) for value in u])
        assert np.allclose(flux.f_array(u), exact)


def test_lipschitz_bound_for_burgers():
    assert lipschitz_bound(Burgers(), -1, 2) == pytest.approx(2, rel=1e-3)


def test_initial_averages_are_exact():
    solver = GodunovSolver(Burgers(), cells=4, window=(0, 1))
    data = PiecewiseConstant((Fraction(3, 8),), (0, 1))
    # The jump at 3/8 splits the second cell [1/4, 1/2] in half.
    assert solver.initial_averages(data).tolist() == [0.0, 0.5, 1.0, 1.0]


def test_constant_data_is_preserved():
    solver = GodunovSolver(Burgers(), cells=20, window=(-1, 1))
    data = PiecewiseConstant((), (Fraction(3, 4),))
    result = solver.solve(data, 1)
    assert np.all(result.averages == 0.75)


def test_lands_exactly_on_the_final_time():
    result = GodunovSolver(Burgers(), cells=50, window=(-1, 2)).solve(
        riemann_data(1, 0), 1
    )
    assert result.t == 1.0
    # dt = 0.9 dx / 1 with dx = 0.06 needs ceil(1 / 0.054) = 19 steps.
    assert result.steps == 19


def test_mass_is_conserved_while_waves_stay_inside():
    data = PiecewiseConstant((0, 1), (0, 1, 0))
    solver = GodunovSolver(Burgers(), cells=200, window=(-1, 3))
    result = solver.solve(data, 1)
    assert result.mass() == pytest.approx(1.0, abs=1e-12)


def test_scheme_is_monotone():
    data = PiecewiseConstant((0, 1), (0, 1, 0))
    result = GodunovSolver(Burgers(), cells=200, window=(-1, 3)).solve(data, 1)
    assert result.averages.min() >= -1e-14
    assert result.averages.max() <= 1 + 1e-14


def test_shock_converges_at_first_order():
    reference = exact_riemann_solution(1, 0)
    errors = [
        GodunovSolver(Burgers(), cells, (-1, 2)).solve(riemann_data(1, 0), 1)
        .l1_error(reference)
        for cells in (100, 200, 400)
    ]
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all(rates > 0.9)


def test_box_data_converges_through_the_interaction():
    # At t = 3 the fan has caught the shock (at t = 2).
    reference = exact_box_solution()
    data = PiecewiseConstant((0, 1), (0, 1, 0))
    coarse, fine = (
        GodunovSolver(Burgers(), cells, (-1, 4)).solve(data, 3).l1_error(reference)
        for cells in (100, 400)
    )
    assert fine < coarse / 2.5


def test_agrees_with_front_tracking_on_buckley_leverett():
    # No closed form here, so the two methods are checked against each other.
    flux = BuckleyLeverett()
    data = riemann_data(1, 0)
    tracked = FrontTracker(flux, Fraction(1, 64)).solve(data, 1)
    distances = [
        GodunovSolver(flux, cells, (-1, 3)).solve(data, 1).l1_error(tracked.evaluate)
        for cells in (100, 400)
    ]
    assert distances[1] < distances[0] / 2
    assert distances[1] < 0.03


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        GodunovSolver(Burgers(), cells=0, window=(0, 1))
    with pytest.raises(ValueError):
        GodunovSolver(Burgers(), cells=10, window=(0, 1), cfl=1.5)
    with pytest.raises(ValueError):
        GodunovSolver(Burgers(), cells=10, window=(1, 0))
