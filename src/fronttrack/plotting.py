"""Plots of a front-tracking solution.

Two views are useful: the fronts as straight segments in the x-t plane, which
shows the whole wave structure and its interactions at a glance, and snapshots
of ``u`` at fixed times.  Nothing here is needed by the solver; the figures of
the documentation are produced from these functions.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .rational import as_fraction
from .solution import Solution

SHOCK_STYLE = {"color": "#c0392b", "linewidth": 1.6}
RAREFACTION_STYLE = {"color": "#2471a3", "linewidth": 0.9}
INTERACTION_STYLE = {"color": "black", "marker": "o", "markersize": 4,
                     "linestyle": "none", "zorder": 3}


def _is_shock(solution: Solution, front, samples=7) -> bool:
    """Whether the front survives the limit ``delta -> 0`` as a shock.

    Oleinik's chord condition is applied to the *original* flux rather than
    to its polygonal approximation: a front is a shock when the true flux
    stays on the admissible side of the chord, and a rarefaction front -- one
    of the small steps making up a fan -- when it does not.  Testing the
    original flux is what distinguishes the two for a nonconvex flux, where a
    rarefaction fan may well run downwards.
    """
    base = solution.flux.base
    low, high = min(front.uL, front.uR), max(front.uL, front.uR)
    decreasing = front.uL > front.uR
    for step in range(1, samples):
        u = low + (high - low) * Fraction(step, samples)
        chord = base.f(front.uL) + front.speed * (u - front.uL)
        if decreasing and base.f(u) > chord:
            return False
        if not decreasing and base.f(u) < chord:
            return False
    return True


def plot_fronts(solution: Solution, ax, xlim=None, show_interactions=True):
    """Draw the fronts of ``solution`` as segments in the x-t plane."""
    for front in solution.fronts:
        style = SHOCK_STYLE if _is_shock(solution, front) else RAREFACTION_STYLE
        for (x0, t0), (x1, t1) in solution.domain.split_segment(
            front.endpoints(solution.t_final)
        ):
            ax.plot([float(x0), float(x1)], [float(t0), float(t1)], **style)
    if show_interactions and solution.interactions:
        ax.plot(
            [float(event.position) for event in solution.interactions],
            [float(event.time) for event in solution.interactions],
            **INTERACTION_STYLE,
        )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.set_ylim(0, float(solution.t_final))
    if xlim is not None:
        ax.set_xlim(*(float(value) for value in xlim))
    return ax


def plot_snapshot(solution: Solution, t, ax, xlim, reference=None, **kwargs):
    """Draw ``u(., t)`` as a staircase, optionally against a reference."""
    t = as_fraction(t)
    state = solution.state_at(t)
    a, b = (as_fraction(value) for value in xlim)
    edges = [a] + [x for x in state.breakpoints if a < x < b] + [b]
    # One connected line, so that the staircase gets one colour and one
    # legend entry; the vertical pieces are the jumps.
    xs, ys = [], []
    for left, right in zip(edges, edges[1:]):
        value = float(state((left + right) / 2))
        xs += [float(left), float(right)]
        ys += [value, value]
    ax.plot(xs, ys, **kwargs)
    if reference is not None:
        grid = np.linspace(float(a), float(b), 601)
        ax.plot(
            grid,
            [float(reference(Fraction(str(x)), t)) for x in grid],
            color="black", linewidth=0.8, linestyle="--", label="exact",
        )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_xlim(float(a), float(b))
    return ax
