"""Weak solutions that are not entropy solutions.

Weak solutions are not unique (Lecture 8&9, section 5): for Burgers' equation
with ``u_l < u_r`` the single jump moving at the Rankine-Hugoniot speed
satisfies the weak formulation just as well as the rarefaction fan does.  The
entropy condition is what rules it out.

:func:`expansion_shock` builds that non-entropic solution as an ordinary
:class:`~fronttrack.solution.Solution`, so every diagnostic of the package can
be run on it: Rankine-Hugoniot and the weak form pass, Oleinik and Kruzkov
fail.  It is also the limit of the Murman-Roe scheme
(:class:`~fronttrack.finite_volume.RoeSolver`) on a transonic rarefaction.
"""

from __future__ import annotations

from fractions import Fraction

from .entropy import shock_speed
from .flux import Burgers, Flux, PolygonalFlux, uniform_nodes
from .fronts import Front
from .initial_data import riemann_data
from .rational import as_fraction
from .solution import Solution


def expansion_shock(uL, uR, t_final, flux: Flux | None = None, delta="1/4",
                    x0=0) -> Solution:
    """The Riemann data joined by one jump at the Rankine-Hugoniot speed.

    For ``u_l < u_r`` and a convex flux this is a weak solution that violates
    the entropy condition.  The polygonal flux on a ``delta`` grid is attached
    only so that the diagnostics can evaluate Oleinik's and Kruzkov's
    conditions against the same nodes the front tracker would use.
    """
    flux = flux if flux is not None else Burgers()
    uL, uR = as_fraction(uL), as_fraction(uR)
    delta = as_fraction(delta)
    nodes = uniform_nodes(min(uL, uR), max(uL, uR), delta, extra=(uL, uR))
    polygonal = PolygonalFlux(flux, nodes)
    data = riemann_data(uL, uR, x0)
    fronts = ()
    if uL != uR:
        front = Front(identifier=0, x0=as_fraction(x0), t0=Fraction(0),
                      speed=shock_speed(uL, uR, polygonal), uL=uL, uR=uR)
        front.t_end = as_fraction(t_final)
        fronts = (front,)
    return Solution(flux=polygonal, initial_data=data, fronts=fronts,
                    interactions=(), t_final=as_fraction(t_final))
