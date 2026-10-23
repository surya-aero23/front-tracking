from fractions import Fraction

import pytest

from fronttrack.diagnostics import check_all_fronts, mass_defect
from fronttrack.flux import Burgers
from fronttrack.initial_data import PiecewiseConstant, riemann_data
from fronttrack.solver import FrontTracker

QUARTER = Fraction(1, 4)


@pytest.fixture
def tracker():
    return FrontTracker(Burgers(), QUARTER)


def test_constant_data_produces_no_fronts(tracker):
    solution = tracker.solve(PiecewiseConstant((), (1,)), 1)
    assert solution.n_fronts == 0
    assert solution.state_at(1).values == (Fraction(1),)


def test_single_shock_travels_at_the_rankine_hugoniot_speed(tracker):
    solution = tracker.solve(riemann_data(1, 0), 2)
    assert solution.n_fronts == 1
    assert solution.n_interactions == 0
    front = solution.fronts[0]
    assert front.speed == Fraction(1, 2)
    assert front.position(2) == 1
    assert solution.evaluate(Fraction(9, 10), 2) == 1
    assert solution.evaluate(Fraction(11, 10), 2) == 0


def test_rarefaction_is_a_fan_of_delta_sized_steps(tracker):
    solution = tracker.solve(riemann_data(0, 1), 1)
    assert solution.n_fronts == 4
    assert solution.n_interactions == 0
    state = solution.state_at(1)
    # The staircase approximates the exact fan u = x/t to within delta.
    for x, value in zip(state.breakpoints, state.values[1:]):
        assert abs(value - x) <= QUARTER


def test_two_shocks_merge_into_one(tracker):
    solution = tracker.solve(PiecewiseConstant((0, 1), (2, 1, 0)), 3)
    assert solution.n_interactions == 1
    event = solution.interactions[0]
    # Speeds 3/2 and 1/2 starting one unit apart meet at t = 1, x = 3/2.
    assert (event.time, event.position) == (Fraction(1), Fraction(3, 2))
    assert len(event.incoming) == 2 and len(event.outgoing) == 1
    merged = solution.fronts[event.outgoing[0]]
    assert (merged.uL, merged.uR, merged.speed) == (2, 0, Fraction(1))
    assert solution.state_at(3).breakpoints == (Fraction(7, 2),)


def test_three_fronts_meeting_at_one_point_are_one_interaction(tracker):
    # Shock speeds 5/2, 3/2 and 1/2 placed so that all three reach the
    # origin at t = 1.  Exact arithmetic sees this as a single collision.
    data = PiecewiseConstant((Fraction(-5, 2), Fraction(-3, 2), Fraction(-1, 2)),
                             (3, 2, 1, 0))
    solution = FrontTracker(Burgers(), QUARTER).solve(data, 2)
    assert solution.n_interactions == 1
    event = solution.interactions[0]
    assert (event.time, event.position) == (Fraction(1), Fraction(0))
    assert len(event.incoming) == 3
    assert solution.fronts[event.outgoing[0]].speed == Fraction(3, 2)


def test_shock_absorbs_a_rarefaction_fan_front_by_front(tracker):
    # Box data: a fan at x = 0 and a shock at x = 1.  The fan fronts are
    # faster than the shock and are swallowed one at a time.
    solution = tracker.solve(PiecewiseConstant((0, 1), (0, 1, 0)), 30)
    assert [event.time for event in solution.interactions] == [
        Fraction(8, 3), Fraction(16, 3), Fraction(16)
    ]
    # The first collision is the fastest fan front meeting the shock.
    assert solution.interactions[0].position == Fraction(7, 3)
    assert solution.state_at(30).values == (Fraction(0), QUARTER, Fraction(0))


def test_interactions_are_recorded_in_time_order(tracker):
    solution = tracker.solve(PiecewiseConstant((0, 1, 2), (3, 2, 1, 0)), 10)
    times = [event.time for event in solution.interactions]
    assert times == sorted(times)


def test_every_front_of_every_run_is_admissible(tracker):
    runs = [
        (riemann_data(1, -1), 2),
        (riemann_data(-1, 1), 2),
        (PiecewiseConstant((0, 1), (0, 1, 0)), 20),
        (PiecewiseConstant((0, 1), (2, 1, 0)), 5),
        (PiecewiseConstant((-1, 0, 1), (1, -1, 1, -1)), 8),
    ]
    for data, t_final in runs:
        solution = tracker.solve(data, t_final)
        assert all(report.ok for report in check_all_fronts(solution))


def test_total_variation_does_not_increase(tracker):
    data = PiecewiseConstant((-1, 0, 1), (0, 2, -1, 1))
    solution = tracker.solve(data, 10)
    times = [Fraction(k, 4) for k in range(41)]
    variations = [solution.total_variation(t) for t in times]
    assert all(b <= a for a, b in zip(variations, variations[1:]))


def test_mass_is_conserved_exactly(tracker):
    data = PiecewiseConstant((0, 1), (0, 1, 0))
    solution = tracker.solve(data, 10)
    for t in [Fraction(0), Fraction(5, 2), Fraction(8, 3), Fraction(10)]:
        assert mass_defect(solution, t, -60, 60) == 0


def test_fronts_stay_in_spatial_order(tracker):
    data = PiecewiseConstant((-1, 0, 1), (2, 0, 2, 0))
    solution = tracker.solve(data, 12)
    for k in range(49):
        t = Fraction(k, 4)
        positions = [front.position(t) for front in solution.active_fronts(t)]
        assert positions == sorted(positions)


def test_solving_to_time_zero_keeps_the_initial_data(tracker):
    data = PiecewiseConstant((0, 1), (2, 1, 0))
    solution = tracker.solve(data, 0)
    assert solution.state_at(0).values == data.values


def test_times_outside_the_interval_are_rejected(tracker):
    solution = tracker.solve(riemann_data(1, 0), 1)
    with pytest.raises(ValueError):
        solution.state_at(2)


def test_delta_must_be_positive():
    with pytest.raises(ValueError):
        FrontTracker(Burgers(), 0)


def test_a_collision_exactly_at_the_final_time_is_counted_once():
    # Shocks of speed 3/2 and 1/2 leave x = 0 and x = 1 and meet at t = 1.
    data = PiecewiseConstant((0, 1), (2, 1, 0))
    solution = FrontTracker(Burgers(), Fraction(1, 4)).solve(data, 1)
    assert solution.n_interactions == 1
    state = solution.state_at(1)
    assert state.breakpoints == (Fraction(3, 2),)
    assert state.values == (2, 0)
