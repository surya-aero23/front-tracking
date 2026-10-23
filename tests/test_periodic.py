from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fronttrack import (
    BuckleyLeverett,
    Burgers,
    FrontTracker,
    Periodic,
    PiecewiseConstant,
    check_all_fronts,
    l1_distance,
)

QUARTER = Fraction(1, 4)


def periodic_tv(state: PiecewiseConstant) -> Fraction:
    """Total variation over one period, the seam jump included."""
    return state.total_variation() + abs(state.values[0] - state.values[-1])


def unrolled(data: PiecewiseConstant, domain: Periodic, copies: int) -> PiecewiseConstant:
    """The periodic data repeated ``copies`` times on each side, on the real line."""
    jumps = []
    for k in range(-copies, copies + 1):
        shift = k * domain.length
        jumps.append((domain.start + shift, data.values[0]))
        jumps += [(x + shift, v) for x, v in zip(data.breakpoints, data.values[1:])]
    return PiecewiseConstant.from_jumps(data.values[-1], jumps)


@st.composite
def periodic_data(draw, length=2):
    cells = 4 * length
    positions = draw(st.lists(st.integers(1, cells - 1), max_size=6, unique=True))
    breakpoints = [Fraction(x, 4) for x in sorted(positions)]
    values = draw(st.lists(st.integers(-6, 6), min_size=len(breakpoints) + 1,
                           max_size=len(breakpoints) + 1))
    return PiecewiseConstant(breakpoints, [v * QUARTER for v in values])


@settings(max_examples=80, deadline=None)
@given(periodic_data())
def test_periodic_solution_matches_the_unrolled_real_line(data):
    domain, t_final = Periodic(0, 2), Fraction(3)
    periodic = FrontTracker(Burgers(), QUARTER, domain).solve(data, t_final)
    # Waves move at most 3/2 per unit time, so 3 copies each side shield the
    # central period up to t = 3.
    line = FrontTracker(Burgers(), QUARTER).solve(unrolled(data, domain, 3), t_final)
    for t in [Fraction(k, 4) for k in range(13)]:
        a, b = periodic.state_at(t), line.state_at(t)
        window = (domain.start, domain.end)
        assert l1_distance(a, b, window) == 0


@settings(max_examples=60, deadline=None)
@given(periodic_data())
def test_mass_is_conserved_and_tv_decays_on_the_ring(data):
    domain = Periodic(0, 2)
    solution = FrontTracker(Burgers(), QUARTER, domain).solve(data, 4)
    assert all(report.ok for report in check_all_fronts(solution))
    times = [Fraction(k, 2) for k in range(9)]
    masses = [solution.state_at(t).mass(0, 2) for t in times]
    assert len(set(masses)) == 1
    variations = [periodic_tv(solution.state_at(t)) for t in times]
    assert all(a >= b for a, b in zip(variations, variations[1:]))


def test_shocks_merge_across_the_seam():
    # Period [0, 2): a shock 1 -> 0 sits on the seam and a shock 2 -> 1 at 3/2
    # catches it from behind, across the seam, at t = 1/2 and x = 1/4.
    data = PiecewiseConstant((Fraction(1, 2), Fraction(3, 2)), (0, 2, 1))
    solution = FrontTracker(Burgers(), QUARTER, Periodic(0, 2)).solve(data, Fraction(1, 2))
    seam_events = [e for e in solution.interactions if e.time == Fraction(1, 2)]
    assert [(e.position, len(e.incoming)) for e in seam_events] == [(QUARTER, 2)]
    state = solution.state_at(Fraction(1, 2))
    assert state(QUARTER - Fraction(1, 100)) == 2
    assert state(QUARTER) == 0


def test_a_whole_ring_can_annihilate():
    # Buckley-Leverett fronts 1/8 -> 1/4 -> 0 -> 1/8 meet at one point; on a
    # ring with nothing else they are all the fronts there are, and they meet
    # exactly on the seam (x = 4 = 0).
    shift = 4
    data = PiecewiseConstant(
        (Fraction(-640, 561) + shift, Fraction(-8, 11) + shift, Fraction(-16, 51) + shift),
        (Fraction(1, 8), QUARTER, 0, Fraction(1, 8)),
    )
    domain = Periodic(0, 4)
    solution = FrontTracker(BuckleyLeverett(), Fraction(1, 8), domain).solve(data, 2)
    (event,) = solution.interactions
    assert (event.time, event.position) == (1, 0)
    assert event.is_cancellation
    assert solution.state_at(2).values == (Fraction(1, 8),)


def test_data_jumping_at_the_seam():
    # u = 1 on (0, 1) and 0 on (1, 2): the seam carries a jump 0 -> 1, a fan.
    data = PiecewiseConstant((1,), (1, 0))
    solution = FrontTracker(Burgers(), QUARTER, Periodic(0, 2)).solve(data, 0)
    assert len(solution.active_fronts(0)) == 1 + 4
    assert solution.state_at(0) == data


def test_fronts_leaving_through_one_end_come_back_at_the_other():
    # With delta = 1 the block 1 on (1/2, 1) is bounded by two fronts of
    # speed 1/2, so it translates rigidly: at t = 5/2 it straddles the seam.
    data = PiecewiseConstant((Fraction(1, 2), 1), (0, 1, 0))
    solution = FrontTracker(Burgers(), 1, Periodic(0, 2)).solve(data, Fraction(5, 2))
    assert solution.n_interactions == 0
    state = solution.state_at(Fraction(5, 2))
    assert state.breakpoints == (QUARTER, Fraction(7, 4))
    assert state.values == (1, 0, 1)
    for (x0, _), (x1, _) in solution.segments():
        assert 0 <= x0 <= 2 and 0 <= x1 <= 2


def test_breakpoints_must_be_inside_the_period():
    with pytest.raises(ValueError):
        FrontTracker(Burgers(), QUARTER, Periodic(0, 2)).solve(
            PiecewiseConstant((2,), (0, 1)), 1)
    with pytest.raises(ValueError):
        Periodic(0, 0)
