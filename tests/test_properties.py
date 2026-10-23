"""Property-based checks of the theorems behind the method.

Each test draws random piecewise constant data and checks one statement from
the lectures (or from Holden and Risebro) on the computed solution.  Data
values lie on the grid ``delta Z``, so that two solutions with different data
still run on the same polygonal flux -- contraction and comparison are
statements about one flux.
"""

from fractions import Fraction

from hypothesis import given, settings
from hypothesis import strategies as st

from fronttrack import (
    BuckleyLeverett,
    Burgers,
    FrontTracker,
    PiecewiseConstant,
    check_all_fronts,
    exact_riemann_solution,
    l1_distance,
    l1_error,
    riemann_data,
)

DELTA = Fraction(1, 4)
T_FINAL = Fraction(6)
SETTINGS = settings(max_examples=60, deadline=None)


@st.composite
def piecewise_constant(draw, max_jumps=8, low=-8, high=8):
    """Random data with breakpoints on (1/4)Z and values on DELTA * Z."""
    positions = draw(
        st.lists(st.integers(-24, 24), min_size=0, max_size=max_jumps, unique=True)
    )
    breakpoints = [Fraction(x, 4) for x in sorted(positions)]
    values = draw(
        st.lists(st.integers(low, high), min_size=len(breakpoints) + 1,
                 max_size=len(breakpoints) + 1)
    )
    return PiecewiseConstant(breakpoints, [v * DELTA for v in values])


@st.composite
def perturbed_pair(draw, nonnegative=False):
    """Data ``u`` and ``v`` that differ only on bounded cells.

    The outer two cells are shared, so that ``u - v`` is integrable.  With
    ``nonnegative`` the perturbation is upwards, so that ``u <= v``.
    """
    u = draw(piecewise_constant(max_jumps=8))
    low = 0 if nonnegative else -4
    shifts = draw(
        st.lists(st.integers(low, 4), min_size=len(u.values), max_size=len(u.values))
    )
    shifts[0] = shifts[-1] = 0
    v = PiecewiseConstant(u.breakpoints, [a + s * DELTA for a, s in zip(u.values, shifts)])
    return u, v


def solve(data, flux=None, t_final=T_FINAL):
    return FrontTracker(flux or Burgers(), DELTA).solve(data, t_final)


TIMES = [Fraction(k, 2) for k in range(13)]


@SETTINGS
@given(perturbed_pair())
def test_l1_contraction(pair):
    """Kruzkov: ``||u(t) - v(t)||_1`` never increases."""
    u, v = (solve(data) for data in pair)
    distances = [l1_distance(u.state_at(t), v.state_at(t)) for t in TIMES]
    assert all(a >= b for a, b in zip(distances, distances[1:]))


@SETTINGS
@given(perturbed_pair(nonnegative=True))
def test_comparison_principle(pair):
    """``u0 <= v0`` implies ``u(t) <= v(t)`` everywhere."""
    u, v = (solve(data) for data in pair)
    for t in TIMES:
        a, b = u.state_at(t), v.state_at(t)
        points = sorted(set(a.breakpoints) | set(b.breakpoints))
        samples = [points[0] - 1] if points else [Fraction(0)]
        samples += [(x + y) / 2 for x, y in zip(points, points[1:])]
        samples += [points[-1] + 1] if points else []
        assert all(a(x) <= b(x) for x in samples)


@SETTINGS
@given(piecewise_constant())
def test_maximum_principle(data):
    solution = solve(data)
    for t in TIMES:
        values = solution.state_at(t).values
        assert data.u_min <= min(values) and max(values) <= data.u_max


@SETTINGS
@given(perturbed_pair())
def test_finite_speed_of_propagation(pair):
    """A change of the data on ``[a, b]`` is felt only on ``[a - St, b + St]``."""
    u_data, v_data = pair
    changed = [x for x, p, q in zip(u_data.breakpoints, u_data.values, v_data.values)
               if p != q] + [x for x, p, q in zip(u_data.breakpoints, u_data.values[1:],
                                                 v_data.values[1:]) if p != q]
    if not changed:
        return
    a, b = min(changed), max(changed)
    speed = max(abs(u_data.u_min), abs(u_data.u_max), abs(v_data.u_min), abs(v_data.u_max))
    u, v = solve(u_data), solve(v_data)
    for t in TIMES:
        far = 100 + speed * t
        left = (a - speed * t - far, a - speed * t)
        right = (b + speed * t, b + speed * t + far)
        for window in (left, right):
            assert l1_distance(u.state_at(t), v.state_at(t), window) == 0


@SETTINGS
@given(piecewise_constant())
def test_discrete_oleinik_condition(data):
    """For Burgers every upward jump at ``t > 0`` is a single fan step."""
    solution = solve(data)
    for t in TIMES[1:]:
        values = solution.state_at(t).values
        assert all(b - a <= DELTA for a, b in zip(values, values[1:]))


@SETTINGS
@given(piecewise_constant())
def test_burgers_interactions_never_increase_the_front_count(data):
    """For a convex flux a collision leaves at most one front behind.

    So the number of fronts drops at every interaction, which bounds the
    number of interactions by the number of initial fronts.
    """
    solution = solve(data)
    assert all(len(event.outgoing) <= 1 for event in solution.interactions)
    initial = len(solution.active_fronts(0))
    assert solution.n_interactions <= initial


@SETTINGS
@given(piecewise_constant())
def test_every_front_is_admissible_and_tv_decays(data):
    solution = solve(data)
    assert all(report.ok for report in check_all_fronts(solution))
    variations = [solution.total_variation(t) for t in TIMES]
    assert all(a >= b for a, b in zip(variations, variations[1:]))


@SETTINGS
@given(piecewise_constant(low=0, high=4))
def test_nonconvex_flux_keeps_entropy_and_maximum_principle(data):
    solution = solve(data, BuckleyLeverett(), t_final=Fraction(2))
    assert all(report.ok for report in check_all_fronts(solution))
    for t in TIMES[:5]:
        values = solution.state_at(t).values
        assert data.u_min <= min(values) and max(values) <= data.u_max


@SETTINGS
@given(st.integers(-8, 8), st.integers(-8, 8), st.sampled_from([1, 2, 4]))
def test_error_obeys_the_flux_perturbation_bound(left, right, t):
    """``||u_delta - u||_1 <= t * Lip(f - f_delta) * TV(u0)`` with Lip <= delta/2.

    This is Kruzkov's stability with respect to the flux; for Burgers the
    slope of ``f - f_delta`` on a cell of width delta is at most delta/2.
    """
    uL, uR = left * DELTA, right * DELTA
    solution = solve(riemann_data(uL, uR), t_final=t)
    spread = max(abs(uL), abs(uR)) * t + 1
    error = l1_error(solution, exact_riemann_solution(uL, uR), t, (-spread, spread),
                     extra_breaks=[uL * t, uR * t])
    bound = t * DELTA / 2 * abs(uR - uL)
    assert error <= float(bound) + 1e-9
