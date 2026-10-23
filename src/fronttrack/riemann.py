"""The Riemann solver for a polygonal flux.

For a piecewise linear flux the entropy solution of a Riemann problem is a fan
of finitely many discontinuities.  Following the recipe of Lecture 8&9,
section 5.16, the states are the vertices of the appropriate envelope and each
front moves at the Rankine-Hugoniot speed, which is the slope of the
corresponding envelope segment.  The speeds come out strictly increasing from
left to right, so the fan is self-consistent and its fronts never collide with
one another.
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

    if uL < uR:
        vertices = lower_convex_envelope(flux.graph_in(uL, uR))
    else:
        vertices = upper_concave_envelope(flux.graph_in(uR, uL))
        vertices.reverse()

    waves = []
    for (u_left, f_left), (u_right, f_right) in zip(vertices, vertices[1:]):
        speed = (f_right - f_left) / (u_right - u_left)
        waves.append(Wave(uL=u_left, uR=u_right, speed=speed))
    return waves
