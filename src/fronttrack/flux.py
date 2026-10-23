"""Flux functions and their piecewise linear (polygonal) approximations.

The solver is written against the small :class:`Flux` interface, so that
Burgers' equation is only the default and not a built-in assumption.  Front
tracking itself always runs on a :class:`PolygonalFlux`: replacing ``f`` by a
piecewise linear interpolant is what turns rarefaction waves into finitely
many discontinuities, so that the whole solution stays piecewise constant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from fractions import Fraction

import numpy as np

from .rational import as_fraction


class Flux(ABC):
    """A scalar flux ``f(u)`` evaluated in exact rational arithmetic."""

    name: str = "flux"
    #: The states where ``f`` has a local extremum.  The Godunov flux of the
    #: finite-volume comparison scheme only needs ``f`` at these points and at
    #: the two cell states.
    extrema: tuple = ()

    @abstractmethod
    def f(self, u) -> Fraction:
        """The flux value ``f(u)``."""

    def df(self, u) -> Fraction:
        """The characteristic speed ``f'(u)``.

        Only needed for exact reference solutions, not by the solver itself.
        """
        raise NotImplementedError(f"{type(self).__name__} has no derivative")

    def __call__(self, u) -> Fraction:
        return self.f(u)

    def f_array(self, u: np.ndarray) -> np.ndarray:
        """``f`` on an array of floats, for the finite-volume scheme.

        The default goes through the exact :meth:`f`, which is slow;
        subclasses override it with a vectorised formula.
        """
        return np.vectorize(lambda value: float(self.f(value)))(u)

    def df_array(self, u: np.ndarray) -> np.ndarray:
        """``f'`` on an array of floats, for the numerical exact solutions.

        The default is a central difference of :meth:`f_array`; subclasses
        override it with the closed form.
        """
        u = np.asarray(u, dtype=float)
        step = 1e-6 * np.maximum(1.0, np.abs(u))
        return (self.f_array(u + step) - self.f_array(u - step)) / (2 * step)


class Burgers(Flux):
    """Burgers' flux ``f(u) = u^2 / 2``; strictly convex."""

    name = "burgers"
    extrema = (0,)

    def f(self, u) -> Fraction:
        u = as_fraction(u)
        return u * u / 2

    def df(self, u) -> Fraction:
        return as_fraction(u)

    def f_array(self, u: np.ndarray) -> np.ndarray:
        return 0.5 * u * u

    def df_array(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(u, dtype=float)


class ConcaveBurgers(Flux):
    """``f(u) = -u^2 / 2``; strictly concave, so the wave rules are reversed."""

    name = "concave-burgers"
    extrema = (0,)

    def f(self, u) -> Fraction:
        u = as_fraction(u)
        return -u * u / 2

    def df(self, u) -> Fraction:
        return -as_fraction(u)

    def f_array(self, u: np.ndarray) -> np.ndarray:
        return -0.5 * u * u

    def df_array(self, u: np.ndarray) -> np.ndarray:
        return -np.asarray(u, dtype=float)


class BuckleyLeverett(Flux):
    """``f(u) = u^2 / (u^2 + M (1 - u)^2)``, the classical nonconvex example.

    The flux is convex near ``u = 0`` and concave near ``u = 1``, so a single
    Riemann problem can produce a shock and a rarefaction at once.
    """

    name = "buckley-leverett"
    # f'(u) = 2 M u (1 - u) / (u^2 + M (1 - u)^2)^2 vanishes only at 0 and 1.
    extrema = (0, 1)

    def __init__(self, mobility=Fraction(1, 2)):
        self.mobility = as_fraction(mobility)

    def f(self, u) -> Fraction:
        u = as_fraction(u)
        denominator = u * u + self.mobility * (1 - u) ** 2
        if denominator == 0:
            raise ValueError("Buckley-Leverett flux is singular at this state")
        return u * u / denominator

    def df(self, u) -> Fraction:
        u = as_fraction(u)
        denominator = u * u + self.mobility * (1 - u) ** 2
        return 2 * self.mobility * u * (1 - u) / denominator ** 2

    def f_array(self, u: np.ndarray) -> np.ndarray:
        mobility = float(self.mobility)
        return u * u / (u * u + mobility * (1 - u) ** 2)

    def df_array(self, u: np.ndarray) -> np.ndarray:
        mobility = float(self.mobility)
        u = np.asarray(u, dtype=float)
        denominator = u * u + mobility * (1 - u) ** 2
        return 2 * mobility * u * (1 - u) / denominator ** 2


class PolygonalFlux(Flux):
    """The piecewise linear interpolant of ``base`` through a set of nodes.

    This is the flux ``f_delta`` of the front-tracking method.  Riemann
    problems for ``f_delta`` have solutions made of discontinuities only, and
    the intermediate states are always nodes, so the node set is invariant
    under the whole computation.
    """

    def __init__(self, base: Flux, nodes):
        nodes = sorted({as_fraction(node) for node in nodes})
        if len(nodes) < 2:
            raise ValueError("a polygonal flux needs at least two nodes")
        self.base = base
        self.name = f"polygonal({base.name})"
        self.nodes: tuple[Fraction, ...] = tuple(nodes)
        self.values: tuple[Fraction, ...] = tuple(base.f(node) for node in nodes)
        # A piecewise linear function attains its extrema at the nodes.
        self.extrema = self.nodes

    @property
    def u_min(self) -> Fraction:
        return self.nodes[0]

    @property
    def u_max(self) -> Fraction:
        return self.nodes[-1]

    def is_node(self, u) -> bool:
        u = as_fraction(u)
        index = bisect_left(self.nodes, u)
        return index < len(self.nodes) and self.nodes[index] == u

    def f(self, u) -> Fraction:
        u = as_fraction(u)
        if not self.u_min <= u <= self.u_max:
            raise ValueError(f"state {u} lies outside the node range")
        index = bisect_left(self.nodes, u)
        if self.nodes[index] == u:
            return self.values[index]
        left, right = self.nodes[index - 1], self.nodes[index]
        f_left, f_right = self.values[index - 1], self.values[index]
        return f_left + (f_right - f_left) * (u - left) / (right - left)

    def nodes_in(self, a, b) -> tuple[Fraction, ...]:
        """All nodes in the closed interval ``[a, b]`` (with ``a <= b``)."""
        a, b = as_fraction(a), as_fraction(b)
        if a > b:
            raise ValueError("expected a <= b")
        return self.nodes[bisect_left(self.nodes, a):bisect_right(self.nodes, b)]

    def graph_in(self, a, b) -> list[tuple[Fraction, Fraction]]:
        """The points ``(u, f(u))`` for all nodes in ``[a, b]``."""
        a, b = as_fraction(a), as_fraction(b)
        start = bisect_left(self.nodes, a)
        stop = bisect_right(self.nodes, b)
        return list(zip(self.nodes[start:stop], self.values[start:stop]))


def uniform_nodes(u_min, u_max, delta, extra=()) -> list[Fraction]:
    """A grid of spacing ``delta`` covering ``[u_min, u_max]``, plus ``extra``.

    The grid is anchored at multiples of ``delta`` and extended outwards to
    cover the range, so that refining ``delta`` by an integer factor refines
    the grid rather than shifting it.  Every state in ``extra`` is added as a
    node, which is how the initial data is represented without any projection
    error.
    """
    u_min, u_max, delta = (as_fraction(v) for v in (u_min, u_max, delta))
    if delta <= 0:
        raise ValueError("delta must be positive")
    first = (u_min / delta).__floor__()
    last = (u_max / delta).__ceil__()
    nodes = {Fraction(j) * delta for j in range(first, last + 1)}
    nodes.update(as_fraction(value) for value in extra)
    if len(nodes) < 2:
        # Constant data: a polygonal flux still needs an interval of states.
        nodes.add(max(nodes) + delta)
    return sorted(nodes)
