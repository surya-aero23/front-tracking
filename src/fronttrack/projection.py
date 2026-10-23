"""Projection of general initial data onto piecewise constants.

Front tracking needs piecewise constant data whose values are states of the
polygonal flux.  Following Holden and Risebro, a general ``u0`` is projected
in two steps: average it over cells of width ``h``, then round each average
to the nearest multiple of ``delta``.  The method then has two approximation
parameters; its error is ``O(h + delta)`` for smooth data, plus the time
growth of the flux error.

The rounding changes the mass of the data by at most ``delta / 2`` per unit
length.  Rounding is what keeps the number of nodes, and so the number of
fronts per Riemann problem, bounded as ``h -> 0``.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import numpy as np

from .initial_data import PiecewiseConstant
from .rational import as_fraction


def project(u0: Callable, window, cells: int, delta, rule="average",
            quadrature_points=6) -> PiecewiseConstant:
    """Piecewise constant data approximating ``u0`` on ``window``.

    ``u0`` is a vectorised function of ``x`` (floats in, floats out).  The
    window is cut into ``cells`` equal cells; each gets the cell average
    (``rule="average"``, by Gauss-Legendre quadrature) or the midpoint value
    (``rule="midpoint"``) of ``u0``, rounded to the grid ``delta Z``.

    On the real line ``u0`` is taken constant outside the window, so the
    first and last cells extend to infinity.  On a periodic domain, pass the
    period as the window; the result then has the breakpoints strictly inside
    it, as :class:`~fronttrack.domain.Periodic` expects.
    """
    if cells < 1:
        raise ValueError("need at least one cell")
    if rule not in ("average", "midpoint"):
        raise ValueError(f"unknown rule {rule!r}")
    a, b = (as_fraction(value) for value in window)
    if not a < b:
        raise ValueError("the window must have a < b")
    delta = as_fraction(delta)
    if delta <= 0:
        raise ValueError("delta must be positive")

    width = (b - a) / cells
    edges = [a + width * i for i in range(cells + 1)]
    lefts = np.array([float(x) for x in edges[:-1]])
    h = float(width)
    if rule == "midpoint":
        raw = np.asarray(u0(lefts + h / 2), dtype=float)
    else:
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_points)
        points = lefts[:, None] + h / 2 * (1 + nodes[None, :])
        raw = np.asarray(u0(points), dtype=float) @ weights / 2

    levels = np.rint(raw / float(delta)).astype(int)
    values = [delta * int(level) for level in levels]
    jumps = list(zip(edges[1:-1], values[1:]))
    return PiecewiseConstant.from_jumps(values[0], jumps)
