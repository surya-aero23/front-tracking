"""Verification of a computed solution.

Three independent checks are provided, matching the three things the method
claims to deliver:

* every front satisfies Rankine-Hugoniot and the entropy conditions, so the
  solution is a weak entropy solution of the polygonal problem
  (:func:`check_all_fronts`);
* the weak formulation itself holds against smooth test functions
  (:func:`weak_form_residual`);
* the conserved quantity balances the boundary flux (:func:`mass_defect`).

:func:`l1_error` measures the distance to a reference solution, which is how
convergence as ``delta -> 0`` is observed, and :func:`l1_distance` is the
exact distance between two piecewise constant functions, used to check
Kruzkov's ``L^1`` contraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .entropy import kruzkov_admissible, oleinik_admissible, rankine_hugoniot_ok
from .initial_data import PiecewiseConstant
from .rational import as_fraction
from .solution import Solution


@dataclass(frozen=True)
class FrontReport:
    """The verdict of all three admissibility tests on a single front."""

    identifier: int
    uL: Fraction
    uR: Fraction
    speed: Fraction
    rankine_hugoniot: bool
    oleinik: bool
    kruzkov: bool

    @property
    def ok(self) -> bool:
        return self.rankine_hugoniot and self.oleinik and self.kruzkov


def check_all_fronts(solution: Solution) -> list[FrontReport]:
    """Apply the admissibility tests to every front of the solution."""
    flux = solution.flux
    return [
        FrontReport(
            identifier=front.identifier,
            uL=front.uL,
            uR=front.uR,
            speed=front.speed,
            rankine_hugoniot=rankine_hugoniot_ok(front.uL, front.uR, front.speed, flux),
            oleinik=oleinik_admissible(front.uL, front.uR, front.speed, flux),
            kruzkov=kruzkov_admissible(front.uL, front.uR, front.speed, flux),
        )
        for front in solution.fronts
    ]


def mass_defect(solution: Solution, t, a, b) -> Fraction:
    """How far the solution is from conserving mass on the window ``[a, b]``.

    The integral conservation law of Lecture 8&9, equation (1.1), says that
    the mass in a fixed interval changes at the rate of the flux difference at
    its two ends.  The window must be wide enough that no front crosses it, so
    that the end states stay constant; the returned defect is then exactly
    zero for a conservative scheme.
    """
    t = as_fraction(t)
    initial = solution.state_at(0)
    flux_in = solution.flux.f(initial(a))
    flux_out = solution.flux.f(initial(b))
    change = solution.mass(t, a, b) - solution.mass(0, a, b)
    return change - t * (flux_in - flux_out)


def l1_distance(u: PiecewiseConstant, v: PiecewiseConstant, window=None) -> Fraction:
    """The exact ``L^1`` distance between two piecewise constant functions.

    Without a ``window`` the distance is over the whole line, which is finite
    only when the two functions share their states at both infinities.
    """
    if window is None:
        if u.u_left != v.u_left or u.u_right != v.u_right:
            raise ValueError("the functions differ at infinity; give a window")
        points = sorted(set(u.breakpoints) | set(v.breakpoints))
        if not points:
            return Fraction(0)
    else:
        a, b = (as_fraction(value) for value in window)
        points = sorted({a, b} | {x for x in u.breakpoints + v.breakpoints if a < x < b})
    return sum(
        (abs(u((left + right) / 2) - v((left + right) / 2)) * (right - left)
         for left, right in zip(points, points[1:])),
        Fraction(0),
    )


@dataclass(frozen=True)
class TestFunction:
    """A compactly supported test function ``phi(x, t)`` and its derivatives.

    The weak formulation of Lecture 8&9, Definition 3.1, only asks for
    ``phi`` in ``C_c^1``, so a polynomial bump is admissible.  Being
    polynomial on its support, it lets Gauss-Legendre quadrature integrate the
    residual exactly rather than approximately.
    """

    value: object
    dx: object
    dt: object
    x_support: tuple[float, float]
    t_support: tuple[float, float]


def polynomial_bump(x_center=0.0, x_radius=1.0, t_center=0.0, t_radius=1.0,
                    power=4) -> TestFunction:
    """The bump ``(1 - r_x^2)^p (1 - r_t^2)^p``, a ``C^(p-1)`` test function."""
    if power < 2:
        raise ValueError("power must be at least 2 for a C^1 test function")

    def factor(s, center, radius):
        r = (np.asarray(s, dtype=float) - center) / radius
        return np.where(np.abs(r) < 1, (1.0 - np.minimum(r * r, 1.0)) ** power, 0.0)

    def derivative(s, center, radius):
        r = (np.asarray(s, dtype=float) - center) / radius
        inner = 1.0 - np.minimum(r * r, 1.0)
        return np.where(
            np.abs(r) < 1,
            -2.0 * power * r * inner ** (power - 1) / radius,
            0.0,
        )

    return TestFunction(
        value=lambda x, t: factor(x, x_center, x_radius) * factor(t, t_center, t_radius),
        dx=lambda x, t: derivative(x, x_center, x_radius) * factor(t, t_center, t_radius),
        dt=lambda x, t: factor(x, x_center, x_radius) * derivative(t, t_center, t_radius),
        x_support=(x_center - x_radius, x_center + x_radius),
        t_support=(t_center - t_radius, t_center + t_radius),
    )


def weak_form_residual(solution: Solution, test_function: TestFunction,
                       order=24) -> float:
    """The residual of the weak formulation of Lecture 8&9, Definition 3.1.

    Evaluates

        int_0^T int (u phi_t + f(u) phi_x) dx dt + int u_0(x) phi(x, 0) dx,

    which vanishes for a weak solution, provided ``phi`` vanishes at ``t = T``.

    The time integral is split at the interaction times, between which the
    fronts move linearly, and the space integral is split at the fronts; both
    are also split at the edges of the support of ``phi``.  Every piece is
    then polynomial, so the quadrature is exact to round-off and any residual
    worth seeing is a genuine failure of the weak formulation.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    a, b = test_function.x_support
    t_start = max(0.0, test_function.t_support[0])
    t_stop = min(float(solution.t_final), test_function.t_support[1])
    if t_stop <= t_start:
        raise ValueError("the test function has no support in [0, t_final]")

    def space_pieces(state):
        edges = [a] + [float(x) for x in state.breakpoints if a < float(x) < b] + [b]
        return list(zip(edges, edges[1:]))

    def space_integral(t, integrand) -> float:
        state = solution.state_at(t) if integrand == "bulk" else solution.state_at(0)
        t_float = float(t)
        total = 0.0
        for left, right in space_pieces(state):
            if right <= left:
                continue
            mid, half = (left + right) / 2, (right - left) / 2
            points = mid + half * nodes
            value = float(state(Fraction(str(mid))))
            if integrand == "bulk":
                flux_value = float(solution.flux.f(state(Fraction(str(mid)))))
                contribution = (value * test_function.dt(points, t_float)
                                + flux_value * test_function.dx(points, t_float))
            else:
                contribution = value * test_function.value(points, 0.0)
            total += half * float(np.sum(weights * contribution))
        return total

    cuts = {Fraction(str(t_start)), Fraction(str(t_stop))}
    cuts |= {event.time for event in solution.interactions
             if Fraction(str(t_start)) < event.time < Fraction(str(t_stop))}
    times = sorted(cuts)

    space_time = 0.0
    for start, stop in zip(times, times[1:]):
        mid, half = (start + stop) / 2, (stop - start) / 2
        for node, weight in zip(nodes, weights):
            # An exact rational node stays strictly inside the subinterval, so
            # state_at never lands on an interaction time.
            t = mid + half * Fraction(node).limit_denominator(10 ** 12)
            space_time += float(half) * weight * space_integral(t, "bulk")

    initial_term = space_integral(Fraction(0), "initial") if t_start == 0.0 else 0.0
    return space_time + initial_term


@dataclass(frozen=True)
class RunStatistics:
    """What a front-tracking run cost.

    ``max_active`` is the largest number of fronts alive at one time and
    ``denominator_digits`` the length of the largest denominator among the
    front origins, a measure of how much exact arithmetic costs.
    """

    fronts: int
    interactions: int
    max_active: int
    denominator_digits: int


def run_statistics(solution: Solution) -> RunStatistics:
    """Count the fronts, interactions and arithmetic size of a run."""
    events = []
    for front in solution.fronts:
        events.append((front.t0, 1))
        if front.t_end is not None and not front.alive:
            events.append((front.t_end, -1))
    # At a common time deaths come first: the outgoing fronts replace them.
    events.sort(key=lambda event: (event[0], event[1]))
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    digits = max(
        (len(str(max(front.x0.denominator, front.t0.denominator)))
         for front in solution.fronts),
        default=0,
    )
    return RunStatistics(
        fronts=solution.n_fronts,
        interactions=solution.n_interactions,
        max_active=peak,
        denominator_digits=digits,
    )


def l1_error(solution: Solution, reference, t, window, resolution=4000,
             extra_breaks=()) -> float:
    """The ``L^1`` distance to a reference solution at time ``t``.

    ``reference`` is a callable ``u(x, t)``.  The window is subdivided at the
    fronts of the computed solution, and at any ``extra_breaks`` given -- the
    kinks of the reference solution, typically -- and a composite midpoint
    rule is used on each piece.  Passing the reference kinks makes the result
    exact whenever both functions are piecewise linear between them.
    """
    state = solution.state_at(t)
    a, b = (float(value) for value in window)
    interior = {float(x) for x in state.breakpoints} | {float(x) for x in extra_breaks}
    edges = [a] + sorted(x for x in interior if a < x < b) + [b]
    t_float = float(t)
    total = 0.0
    per_piece = max(2, resolution // max(1, len(edges) - 1))
    for left, right in zip(edges, edges[1:]):
        if right <= left:
            continue
        value = float(state((Fraction(str(left)) + Fraction(str(right))) / 2))
        step = (right - left) / per_piece
        points = left + step * (np.arange(per_piece) + 0.5)
        reference_values = np.array(
            [float(reference(Fraction(str(point)), t)) for point in points]
        )
        total += step * np.sum(np.abs(value - reference_values))
    return float(total)
