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


def exact_box_solution(height=1, width=1) -> Callable:
    """The exact Burgers solution for box data ``u0 = height`` on ``[0, width]``.

    A fan opens at ``x = 0`` and a shock of speed ``height / 2`` leaves
    ``x = width``.  The fan's head catches the shock at ``t* = 2 width /
    height``; afterwards the shock sits at ``sqrt(2 height width t)``, as
    follows from conservation of the mass ``height * width``.
    """
    height, width = as_fraction(height), as_fraction(width)
    if height <= 0 or width <= 0:
        raise ValueError("the box must have positive height and width")
    catch_up = 2 * width / height

    def solution(x, t):
        x, t = as_fraction(x), as_fraction(t)
        if t <= 0:
            raise ValueError("the exact solution is defined for t > 0")
        if x <= 0:
            return Fraction(0)
        if t <= catch_up:
            if x < height * t:
                return x / t
            return height if x < width + height * t / 2 else Fraction(0)
        # x/t is the fan's value up to the shock at sqrt(2 height width t),
        # compared here through squares to stay in exact arithmetic.
        return x / t if x * x < 2 * height * width * t else Fraction(0)

    return solution
