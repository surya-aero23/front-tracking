"""Exact solutions, used as references for the convergence tests.

For the quadratic fluxes the Riemann solution has the closed form of Lecture
8&9, sections 5.3 and 5.4 -- a single shock or a centred fan -- and is
evaluated in exact arithmetic.  For any other smooth flux the envelope
construction of section 5.13 is carried out numerically on the true flux:
straight pieces of the envelope are shocks, pieces where it touches the graph
are fans on which ``f'(u) = x/t`` is inverted.  The tangency points, such as
the Welge point of Buckley-Leverett, are refined to machine precision.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .flux import Burgers, ConcaveBurgers, Flux
from .rational import as_fraction


def exact_riemann_solution(uL, uR, flux: Flux | None = None) -> Callable:
    """The exact entropy solution of the Riemann problem as ``u(x, t)``.

    Returns a callable ``u(x, t)`` for ``t > 0``.  For Burgers' flux and its
    concave counterpart the values are exact fractions; for any other flux
    they are floats from :func:`envelope_riemann_waves`.
    """
    flux = flux if flux is not None else Burgers()
    if not isinstance(flux, (Burgers, ConcaveBurgers)):
        return _numerical_riemann_solution(uL, uR, flux)
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


@dataclass(frozen=True)
class ExactWave:
    """One wave of an exact Riemann solution, from ``left`` to ``right``.

    A shock moves at ``speed``; a fan spans the characteristic speeds
    ``f'(left)`` to ``f'(right)``.
    """

    kind: str
    left: float
    right: float
    speed: float | None = None


def envelope_riemann_waves(uL, uR, flux: Flux, samples=20001) -> list[ExactWave]:
    """The waves of the exact Riemann solution of a smooth flux, left to right.

    The lower convex envelope (``u_l < u_r``) or upper concave envelope
    (``u_l > u_r``) is first found on ``samples`` points.  Its straight pieces
    are shocks; each endpoint of a shock that lies inside ``[u_l, u_r]`` is a
    point of tangency, which is then refined by bisection on
    ``f'(p)(p - q) = f(p) - f(q)``.  What remains between shocks are fans.
    """
    uL, uR = float(uL), float(uR)
    if uL == uR:
        return []
    sign = 1.0 if uL < uR else -1.0
    low, high = min(uL, uR), max(uL, uR)

    def g(u):
        return sign * flux.f_array(np.asarray(u, dtype=float))

    def dg(u):
        return sign * flux.df_array(np.asarray(u, dtype=float))

    u = np.linspace(low, high, samples)
    hull = _lower_hull_indices(u, g(u))
    spacing = u[1] - u[0]

    shocks = []
    for i, j in zip(hull, hull[1:]):
        if j - i > 1:
            shocks.append([float(u[i]), float(u[j])])
    for shock in shocks:
        for _ in range(4):
            if low < shock[0]:
                shock[0] = _tangent_point(shock[0], shock[1], g, dg, spacing, low, high)
            if shock[1] < high:
                shock[1] = _tangent_point(shock[1], shock[0], g, dg, spacing, low, high)

    # Pieces in increasing u; fans fill the gaps between shocks.
    pieces, cursor = [], low
    for a, b in shocks:
        if a > cursor:
            pieces.append(("fan", cursor, a))
        pieces.append(("shock", a, b))
        cursor = b
    if cursor < high:
        pieces.append(("fan", cursor, high))
    if sign < 0:
        # For u_l > u_r the solution runs from high states down to low ones.
        pieces = [(kind, b, a) for kind, a, b in reversed(pieces)]

    waves = []
    for kind, a, b in pieces:
        if kind == "shock":
            speed = float((flux.f_array(np.array(b)) - flux.f_array(np.array(a))) / (b - a))
            waves.append(ExactWave("shock", a, b, speed))
        else:
            waves.append(ExactWave("fan", a, b))
    return waves


def _lower_hull_indices(x: np.ndarray, y: np.ndarray) -> list[int]:
    """Indices of the lower convex hull of points sorted by ``x``."""
    hull: list[int] = []
    for k in range(len(x)):
        while len(hull) >= 2:
            i, j = hull[-2], hull[-1]
            cross = (x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i])
            if cross > 0:
                break
            hull.pop()
        hull.append(k)
    return hull


def _tangent_point(p, q, g, dg, spacing, low, high, iterations=80):
    """Refine ``p`` so that the chord from ``q`` is tangent to ``g`` at ``p``."""
    def residual(point):
        return float(dg(point) * (point - q) - (g(point) - g(q)))

    a, b = max(low, p - 2 * spacing), min(high, p + 2 * spacing)
    ra, rb = residual(a), residual(b)
    if ra == 0:
        return a
    if ra * rb > 0:
        return p
    for _ in range(iterations):
        middle = 0.5 * (a + b)
        rm = residual(middle)
        if ra * rm <= 0:
            b = middle
        else:
            a, ra = middle, rm
    return float(0.5 * (a + b))


def _invert_speed(flux: Flux, a, b, xi, iterations=80):
    """The state between ``a`` and ``b`` with ``f'(u) = xi`` (``f'`` monotone)."""
    da = float(flux.df_array(np.array(a)))
    for _ in range(iterations):
        middle = 0.5 * (a + b)
        if (float(flux.df_array(np.array(middle))) - xi) * (da - xi) > 0:
            a = middle
        else:
            b = middle
    return float(0.5 * (a + b))


def _numerical_riemann_solution(uL, uR, flux: Flux) -> Callable:
    waves = envelope_riemann_waves(uL, uR, flux)
    uL, uR = float(uL), float(uR)

    def solution(x, t):
        x, t = float(x), float(t)
        if t <= 0:
            raise ValueError("the exact solution is defined for t > 0")
        xi = x / t
        for wave in waves:
            if wave.kind == "shock":
                if xi < wave.speed:
                    return wave.left
                continue
            first = float(flux.df_array(np.array(wave.left)))
            last = float(flux.df_array(np.array(wave.right)))
            if xi < first:
                return wave.left
            if xi <= last:
                return _invert_speed(flux, wave.left, wave.right, xi)
        return uR

    solution.waves = waves
    return solution


def exact_sine_solution(amplitude=1.0) -> Callable:
    """The entropy solution of Burgers' equation for ``u0 = A sin(2 pi x)``, period 1.

    Characteristics ``x = xi + t A sin(2 pi xi)`` first cross at the breaking
    time ``t_b = 1 / (2 pi A)``.  By the symmetry ``u(1 - x) = -u(x)`` the
    shock that forms sits still at ``x = 1/2``, and for ``0 <= x < 1/2`` the
    solution is carried by the characteristics from ``[0, xi_s]`` that have
    not yet reached it.  Values are floats.
    """
    amplitude = float(amplitude)
    if amplitude <= 0:
        raise ValueError("the amplitude must be positive")

    def foot(x, t, upper):
        # x(xi) is increasing on [0, upper]; invert it by bisection.
        low, high = 0.0, upper
        for _ in range(80):
            middle = 0.5 * (low + high)
            if middle + t * amplitude * np.sin(2 * np.pi * middle) < x:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def last_foot(t):
        """The foot of the characteristic arriving at x = 1/2 from the left."""
        if 2 * np.pi * amplitude * t <= 1:
            return 0.5
        turn = np.arccos(-1 / (2 * np.pi * amplitude * t)) / (2 * np.pi)
        return foot(0.5, t, turn)

    def left_half(x, t):
        """The solution for ``0 <= x <= 1/2``, as the limit from the left."""
        return float(amplitude * np.sin(2 * np.pi * foot(x, t, last_foot(t))))

    def solution(x, t):
        x, t = float(x), float(t)
        if t < 0:
            raise ValueError("the solution is defined for t >= 0")
        x = x % 1.0
        if x < 0.5:
            return left_half(x, t)
        # Odd symmetry about 1/2; at the shock itself take the right value.
        return -left_half(1.0 - x, t)

    solution.breaking_time = 1 / (2 * np.pi * amplitude)
    return solution
