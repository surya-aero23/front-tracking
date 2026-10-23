"""Admissibility tests for a single front.

Three independent conditions from the lectures are available, and the
diagnostics apply all of them to every front of a computed solution:

* the Rankine-Hugoniot condition ``s [u] = [f(u)]`` (Lecture 8&9, Theorem 4.1),
  which is exactly what makes a piecewise constant function with straight
  fronts a weak solution;
* Oleinik's chord condition (Lecture 8&9, section 5.11), the geometric form of
  the entropy condition;
* the Kruzkov jump inequality (Lecture 14, Lemma 2.1), the form that comes
  from the vanishing viscosity limit.

For a piecewise linear flux the last two only have to be checked at the nodes
lying between the two states, since the flux is affine in between.
"""

from __future__ import annotations

from fractions import Fraction

from .flux import PolygonalFlux
from .rational import as_fraction, sign


def shock_speed(uL, uR, flux) -> Fraction:
    """The Rankine-Hugoniot speed of a jump from ``uL`` to ``uR``."""
    uL, uR = as_fraction(uL), as_fraction(uR)
    if uL == uR:
        raise ValueError("a jump needs two distinct states")
    return (flux.f(uR) - flux.f(uL)) / (uR - uL)


def rankine_hugoniot_ok(uL, uR, speed, flux) -> bool:
    """Whether ``speed`` satisfies ``s (uR - uL) = f(uR) - f(uL)`` exactly."""
    uL, uR, speed = (as_fraction(v) for v in (uL, uR, speed))
    return speed * (uR - uL) == flux.f(uR) - flux.f(uL)


def oleinik_admissible(uL, uR, speed, flux: PolygonalFlux) -> bool:
    """Oleinik's chord condition for the jump ``uL -> uR``.

    For a decreasing jump the flux must lie below the chord joining the two
    states, and for an increasing jump it must lie above it.  Both cases are
    the statement that no intermediate state can be split off as a faster
    sub-shock.
    """
    uL, uR, speed = (as_fraction(v) for v in (uL, uR, speed))
    low, high = min(uL, uR), max(uL, uR)
    for node in flux.nodes_in(low, high):
        chord = flux.f(uL) + speed * (node - uL)
        if uL > uR and flux.f(node) > chord:
            return False
        if uL < uR and flux.f(node) < chord:
            return False
    return True


def kruzkov_admissible(uL, uR, speed, flux: PolygonalFlux) -> bool:
    """The Kruzkov jump inequality ``s [|u - k|] >= [sgn(u - k)(f(u) - f(k))]``.

    Tested for every node ``k`` strictly between the two states, which is
    enough for a piecewise linear flux.
    """
    uL, uR, speed = (as_fraction(v) for v in (uL, uR, speed))
    low, high = min(uL, uR), max(uL, uR)
    f_left, f_right = flux.f(uL), flux.f(uR)
    for k in flux.nodes_in(low, high):
        if k in (low, high):
            continue
        f_k = flux.f(k)
        entropy_jump = abs(uR - k) - abs(uL - k)
        entropy_flux_jump = (
            sign(uR - k) * (f_right - f_k) - sign(uL - k) * (f_left - f_k)
        )
        if speed * entropy_jump < entropy_flux_jump:
            return False
    return True


def is_admissible(uL, uR, speed, flux: PolygonalFlux) -> bool:
    """All three conditions at once."""
    return (
        rankine_hugoniot_ok(uL, uR, speed, flux)
        and oleinik_admissible(uL, uR, speed, flux)
        and kruzkov_admissible(uL, uR, speed, flux)
    )
