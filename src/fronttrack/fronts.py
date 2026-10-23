"""Fronts: the straight line discontinuities tracked by the method.

A front carries the two states it separates and the Rankine-Hugoniot speed at
which it travels, and lives on a time interval ``[t0, t_end]`` between the
event that created it and the collision that destroys it.  The ``previous``
and ``next`` links keep the fronts in spatial order, so that the solver can
find the neighbours of a collision in constant time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .rational import as_fraction


@dataclass(eq=False)
class Front:
    """A discontinuity from ``uL`` to ``uR`` moving at constant ``speed``."""

    identifier: int
    x0: Fraction
    t0: Fraction
    speed: Fraction
    uL: Fraction
    uR: Fraction
    t_end: Fraction | None = None
    previous: "Front | None" = field(default=None, repr=False, compare=False)
    next: "Front | None" = field(default=None, repr=False, compare=False)
    alive: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self):
        self.x0 = as_fraction(self.x0)
        self.t0 = as_fraction(self.t0)
        self.speed = as_fraction(self.speed)
        self.uL = as_fraction(self.uL)
        self.uR = as_fraction(self.uR)
        if self.uL == self.uR:
            raise ValueError("a front must separate two distinct states")

    def position(self, t) -> Fraction:
        """Where the front is at time ``t``, extrapolated if need be."""
        return self.x0 + self.speed * (as_fraction(t) - self.t0)

    @property
    def jump(self) -> Fraction:
        return self.uR - self.uL

    def alive_at(self, t, t_final) -> bool:
        """Whether the front carries the solution at time ``t``.

        The interval is taken half open, ``[t0, t_end)``, so that at an
        interaction time only the outgoing fronts count.  A front that
        survives to the final time is the exception: nothing comes after it,
        so its interval is closed.  A front destroyed by a collision exactly
        at the final time is not, since the outgoing fronts replace it there.
        """
        t = as_fraction(t)
        if t < self.t0:
            return False
        if self.t_end is None:
            return True
        survives = self.alive and t == self.t_end == as_fraction(t_final)
        return t < self.t_end or survives

    def endpoints(self, t_final) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
        """The segment ``((x0, t0), (x_end, t_end))`` in the x-t plane."""
        t_end = as_fraction(t_final) if self.t_end is None else self.t_end
        return (self.x0, self.t0), (self.position(t_end), t_end)
