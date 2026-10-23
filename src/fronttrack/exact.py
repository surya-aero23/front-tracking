"""Exact Riemann solutions, used as a reference for the convergence tests.

Only the convex and concave cases are needed, since these are the fluxes for
which the exact solution has the closed form of Lecture 8&9, sections 5.3 and
5.4: either a single shock or a centred rarefaction fan.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from .flux import Burgers, ConcaveBurgers, Flux
from .rational import as_fraction


def exact_riemann_solution(uL, uR, flux: Flux | None = None) -> Callable:
    """The exact entropy solution of the Riemann problem as ``u(x, t)``.

    Returns a callable ``u(x, t)`` for ``t > 0``.  Supported fluxes are
    Burgers' flux and its concave counterpart, where ``f'`` is invertible in
    closed form.
    """
    flux = flux if flux is not None else Burgers()
    if not isinstance(flux, (Burgers, ConcaveBurgers)):
        raise NotImplementedError(
            "exact solutions are only available for the quadratic fluxes"
        )
    uL, uR = as_fraction(uL), as_fraction(uR)
    convex = isinstance(flux, Burgers)
    # f'(u) = u for Burgers and -u for the concave flux, so the characteristic
    # speed of a state is u or -u, and a shock forms when they converge.
    speed_of = (lambda u: u) if convex else (lambda u: -u)
    shock = (uL > uR) if convex else (uL < uR)

    def solution(x, t):
        x, t = as_fraction(x), as_fraction(t)
        if t <= 0:
            raise ValueError("the exact solution is defined for t > 0")
        xi = x / t
        if uL == uR:
            return uL
        if shock:
            speed = (flux.f(uR) - flux.f(uL)) / (uR - uL)
            return uL if xi < speed else uR
        left_speed, right_speed = speed_of(uL), speed_of(uR)
        if xi <= left_speed:
            return uL
        if xi >= right_speed:
            return uR
        return xi if convex else -xi

    return solution
