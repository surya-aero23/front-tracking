"""The Riemann solver for a polygonal flux.

For a piecewise linear flux the entropy solution of a Riemann problem is a fan
of finitely many discontinuities.  Following the recipe of Lecture 8&9,
section 5.16, the states are the vertices of the appropriate envelope and each
front moves at the Rankine-Hugoniot speed, which is the slope of the
corresponding envelope segment.  The speeds come out strictly increasing from
left to right, so the fan is self-consistent and its fronts never collide with
one another.

For a flux known to be convex (or concave) the envelope is known in advance:
the chord for a shock and the whole polygon for a fan (sections 5.3 and 5.4).
The solver uses that shortcut, which turns the cost of a shock from the
number of nodes it spans into a constant; the envelope remains the general
path and the two are checked against each other in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .envelope import lower_convex_envelope, upper_concave_envelope
from .flux import PolygonalFlux
from .rational import as_fraction


@dataclass(frozen=True)
class Wave:
    """A single discontinuity of a Riemann fan, from ``uL`` to ``uR``."""

    uL: Fraction
    uR: Fraction
    speed: Fraction

    @property
    def jump(self) -> Fraction:
        return self.uR - self.uL


def solve_riemann(uL, uR, flux: PolygonalFlux) -> list[Wave]:
    """The entropy solution of the Riemann problem ``(uL, uR)`` for ``flux``.

    Returns the waves ordered from left to right, with strictly increasing
    speeds.  Equal states give an empty list, which is how wave cancellation
    appears in the solver.
    """
    uL, uR = as_fraction(uL), as_fraction(uR)
    for state in (uL, uR):
        if not flux.is_node(state):
            raise ValueError(f"state {state} is not a node of the polygonal flux")
    if uL == uR:
        return []

    vertices = _vertices(uL, uR, flux)

    waves = []
    for (u_left, f_left), (u_right, f_right) in zip(vertices, vertices[1:]):
        speed = (f_right - f_left) / (u_right - u_left)
        waves.append(Wave(uL=u_left, uR=u_right, speed=speed))
    return waves


def _vertices(uL, uR, flux: PolygonalFlux) -> list[tuple[Fraction, Fraction]]:
    """The envelope vertices from ``uL`` to ``uR``, in the order of the fan."""
    shape = flux.convexity if uL < uR else -flux.convexity
    if shape < 0:
        # Convex flux with a decreasing jump, or concave with an increasing
        # one: the envelope is the chord, a single shock.
        return [(uL, flux.f(uL)), (uR, flux.f(uR))]
    if uL < uR:
        graph = flux.graph_in(uL, uR)
        return graph if shape > 0 else lower_convex_envelope(graph)
    graph = flux.graph_in(uR, uL)
    vertices = graph if shape > 0 else upper_concave_envelope(graph)
    return vertices[::-1]
