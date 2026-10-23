"""Finite-volume schemes, used as an independent comparison.

Front tracking computes the exact solution of an approximate problem (the
polygonal flux); a finite-volume scheme computes an approximate solution of the
exact problem.  Comparing the two is a check that does not share any code with
the front tracker except the flux itself.

The scheme is the first-order Godunov method on a uniform grid:

    u_i^{n+1} = u_i^n - (dt / dx) (F(u_i, u_{i+1}) - F(u_{i-1}, u_i)),

with the Godunov flux

    F(a, b) = min_{a <= u <= b} f(u)   if a <= b,
              max_{b <= u <= a} f(u)   if a > b,

which is the flux of the exact Riemann solution at ``x/t = 0``.  It works in
floating point: the scheme is only first order, so exact arithmetic would buy
nothing.

The Murman-Roe scheme is included as a cautionary example: it is
conservative and consistent, but upwinds with the Rankine-Hugoniot speed and so
cannot open a transonic rarefaction.  It converges to the expansion shock, a
weak solution that violates the entropy condition (see
:mod:`fronttrack.counterexamples`).

The real line is truncated to a finite window.  The ghost cells hold the
far-field states of the initial data, which is exact for as long as no wave
reaches the edge of the window -- so the window must be chosen wide enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flux import Flux
from .initial_data import PiecewiseConstant
from .rational import as_fraction


def godunov_flux(flux: Flux, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The Godunov numerical flux at each interface.

    The minimum (or maximum) of ``f`` over the interval between the two states
    is attained either at one of the states or at an interior extremum of
    ``f``, so only those candidates are evaluated.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    f_left, f_right = flux.f_array(left), flux.f_array(right)
    increasing = left <= right
    low = np.minimum(f_left, f_right)
    high = np.maximum(f_left, f_right)
    low_state, high_state = np.minimum(left, right), np.maximum(left, right)
    for extremum in flux.extrema:
        extremum = float(extremum)
        inside = (low_state < extremum) & (extremum < high_state)
        value = float(flux.f_array(np.array(extremum)))
        low = np.where(inside, np.minimum(low, value), low)
        high = np.where(inside, np.maximum(high, value), high)
    return np.where(increasing, low, high)


def roe_flux(flux: Flux, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The Murman-Roe flux: upwind along the Rankine-Hugoniot speed.

    ``F(a, b) = f(a)`` if the chord slope of ``f`` between ``a`` and ``b`` is
    non-negative and ``f(b)`` otherwise.  This has no entropy fix.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    f_left, f_right = flux.f_array(left), flux.f_array(right)
    jump = right - left
    safe = np.where(jump == 0, 1.0, jump)
    # For equal states the chord degenerates; f is then the same either way.
    speed = np.where(jump == 0, 0.0, (f_right - f_left) / safe)
    return np.where(speed >= 0, f_left, f_right)


def lipschitz_bound(flux: Flux, u_min, u_max, samples=4097) -> float:
    """An estimate of ``max |f'|`` on ``[u_min, u_max]`` from sampled chords."""
    u_min, u_max = float(u_min), float(u_max)
    if u_min == u_max:
        u_max = u_min + 1.0
    u = np.linspace(u_min, u_max, samples)
    slopes = np.diff(flux.f_array(u)) / np.diff(u)
    return float(np.max(np.abs(slopes)))


@dataclass(frozen=True)
class FiniteVolumeSolution:
    """Cell averages on a uniform grid at time ``t``."""

    edges: np.ndarray
    averages: np.ndarray
    t: float
    steps: int

    @property
    def dx(self) -> float:
        return float(self.edges[1] - self.edges[0])

    @property
    def centers(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def evaluate(self, x) -> np.ndarray:
        """The piecewise constant reconstruction at the points ``x``."""
        x = np.asarray(x, dtype=float)
        index = np.searchsorted(self.edges, x, side="right") - 1
        index = np.clip(index, 0, len(self.averages) - 1)
        return self.averages[index]

    def mass(self) -> float:
        """The integral of the cell averages over the window."""
        return float(np.sum(self.averages) * self.dx)

    def l1_error(self, reference, samples_per_cell=8) -> float:
        """The ``L^1`` distance over the window to a reference ``u(x, t)``.

        ``reference`` is called pointwise, as the exact solutions and the
        front-tracking :meth:`~fronttrack.solution.Solution.evaluate` are; a
        midpoint rule with ``samples_per_cell`` points is used in each cell.
        """
        offsets = (np.arange(samples_per_cell) + 0.5) / samples_per_cell
        points = self.edges[:-1, None] + self.dx * offsets[None, :]
        t = as_fraction(self.t)
        values = np.array(
            [float(reference(as_fraction(float(x)), t)) for x in points.ravel()]
        ).reshape(points.shape)
        step = self.dx / samples_per_cell
        return float(np.sum(np.abs(values - self.averages[:, None])) * step)


class FiniteVolumeSolver:
    """A conservative three-point scheme for ``u_t + f(u)_x = 0``.

    ``cells`` uniform cells cover ``window = (a, b)``; the time step is
    ``cfl * dx / max|f'|``, shortened at the end to land exactly on
    ``t_final``.  Subclasses choose the numerical flux.
    """

    #: ``numerical_flux(flux, left_states, right_states)`` at every interface.
    numerical_flux = staticmethod(godunov_flux)

    def __init__(self, flux: Flux, cells: int, window, cfl=0.9):
        if cells < 1:
            raise ValueError("need at least one cell")
        if not 0 < cfl <= 1:
            raise ValueError("the CFL number must lie in (0, 1]")
        a, b = (float(value) for value in window)
        if not a < b:
            raise ValueError("the window must have a < b")
        self.flux = flux
        self.cells = cells
        self.window = (a, b)
        self.cfl = cfl

    def initial_averages(self, data: PiecewiseConstant) -> np.ndarray:
        """Exact cell averages of the piecewise constant initial data."""
        a, b = (as_fraction(value) for value in self.window)
        dx = (b - a) / self.cells
        edges = [a + dx * i for i in range(self.cells + 1)]
        return np.array(
            [float(data.mass(left, right) / dx) for left, right in zip(edges, edges[1:])]
        )

    def solve(self, data: PiecewiseConstant, t_final) -> FiniteVolumeSolution:
        t_final = float(t_final)
        if t_final < 0:
            raise ValueError("t_final must be non-negative")
        a, b = self.window
        edges = np.linspace(a, b, self.cells + 1)
        dx = (b - a) / self.cells
        u = self.initial_averages(data)
        # The scheme is monotone, so the solution stays in the range of the data
        # and this bound holds for all time.
        speed = lipschitz_bound(self.flux, data.u_min, data.u_max)
        dt_max = self.cfl * dx / speed if speed > 0 else t_final
        ghost_left, ghost_right = float(data.u_left), float(data.u_right)

        t, steps = 0.0, 0
        while t < t_final:
            dt = min(dt_max, t_final - t)
            padded = np.concatenate(([ghost_left], u, [ghost_right]))
            fluxes = self.numerical_flux(self.flux, padded[:-1], padded[1:])
            u = u - dt / dx * np.diff(fluxes)
            t += dt
            steps += 1
        return FiniteVolumeSolution(edges=edges, averages=u, t=t_final, steps=steps)


class GodunovSolver(FiniteVolumeSolver):
    """The Godunov scheme: monotone, and convergent to the entropy solution."""

    numerical_flux = staticmethod(godunov_flux)


class RoeSolver(FiniteVolumeSolver):
    """The Murman-Roe scheme, without entropy fix.

    It is conservative, so its limits are weak solutions -- but not
    necessarily the entropy solution.
    """

    numerical_flux = staticmethod(roe_flux)
