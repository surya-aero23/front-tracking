from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fronttrack.exact import exact_riemann_solution  # noqa: E402
from fronttrack.flux import BuckleyLeverett, Burgers  # noqa: E402
from fronttrack.initial_data import PiecewiseConstant, riemann_data  # noqa: E402
from fronttrack.plotting import _is_shock, plot_fronts, plot_snapshot  # noqa: E402
from fronttrack.solver import FrontTracker  # noqa: E402

EIGHTH = Fraction(1, 8)


def test_burgers_shock_and_fan_are_told_apart():
    shock = FrontTracker(Burgers(), EIGHTH).solve(riemann_data(1, 0), 1)
    assert _is_shock(shock, shock.fronts[0])
    fan = FrontTracker(Burgers(), EIGHTH).solve(riemann_data(0, 1), 1)
    assert not any(_is_shock(fan, front) for front in fan.fronts)


def test_a_decreasing_nonconvex_fan_is_not_mistaken_for_shocks():
    # Buckley-Leverett resolves u_l = 1 > u_r = 0 into a downward running
    # rarefaction followed by a single shock.
    solution = FrontTracker(BuckleyLeverett(), Fraction(1, 16)).solve(
        riemann_data(1, 0), 1
    )
    flags = [_is_shock(solution, front) for front in solution.fronts]
    assert flags.count(True) == 1
    assert flags[-1] is True


def test_plot_fronts_draws_one_line_per_front():
    solution = FrontTracker(Burgers(), EIGHTH).solve(
        PiecewiseConstant((0, 1), (0, 1, 0)), 12
    )
    _, ax = plt.subplots()
    plot_fronts(solution, ax, xlim=(-1, 8))
    # One line per front, plus one for the interaction markers.
    assert len(ax.lines) == solution.n_fronts + 1
    plt.close("all")


def test_plot_snapshot_draws_the_staircase_and_the_reference():
    solution = FrontTracker(Burgers(), EIGHTH).solve(riemann_data(0, 1), 1)
    _, ax = plt.subplots()
    plot_snapshot(solution, 1, ax, xlim=(-2, 3),
                  reference=exact_riemann_solution(0, 1), label="tracked")
    # The staircase is one line (one colour, one legend entry), plus the
    # reference curve.
    assert len(ax.lines) == 2
    assert ax.get_legend_handles_labels()[1] == ["tracked", "exact"]
    staircase = ax.lines[0]
    assert len(staircase.get_xdata()) == 2 * len(solution.state_at(1).values)
    plt.close("all")
