"""The computed solution: fronts, interactions and the function they define."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .domain import Domain, RealLine
from .flux import PolygonalFlux
from .fronts import Front
from .initial_data import PiecewiseConstant
from .rational import as_fraction


@dataclass(frozen=True)
class Interaction:
    """A collision of fronts at a single point of the x-t plane."""

    time: Fraction
    position: Fraction
    incoming: tuple[int, ...]
    outgoing: tuple[int, ...]

    @property
    def is_cancellation(self) -> bool:
        """Whether the colliding fronts annihilate instead of producing new ones."""
        return not self.outgoing


@dataclass(frozen=True)
class Solution:
    """A front-tracking solution on the time interval ``[0, t_final]``.

    The solution is piecewise constant at every time: :meth:`state_at` returns
    it as a :class:`~fronttrack.initial_data.PiecewiseConstant`, which is also
    how the diagnostics and the plots read it.
    """

    flux: PolygonalFlux
    initial_data: PiecewiseConstant
    fronts: tuple[Front, ...]
    interactions: tuple[Interaction, ...]
    t_final: Fraction
    domain: Domain = field(default_factory=RealLine)

    @property
    def u_left(self) -> Fraction:
        """The constant state at ``x = -infinity``."""
        return self.initial_data.u_left

    @property
    def u_right(self) -> Fraction:
        """The constant state at ``x = +infinity``."""
        return self.initial_data.u_right

    @property
    def n_fronts(self) -> int:
        """How many fronts were created in total."""
        return len(self.fronts)

    @property
    def n_interactions(self) -> int:
        return len(self.interactions)

    def active_fronts(self, t) -> list[Front]:
        """The fronts carrying the solution at time ``t``, in spatial order."""
        t = as_fraction(t)
        if not 0 <= t <= self.t_final:
            raise ValueError(f"time {t} is outside [0, {self.t_final}]")
        active = [front for front in self.fronts if front.alive_at(t, self.t_final)]
        active.sort(key=lambda front: (self.domain.position(front, t), front.speed))
        return active

    def state_at(self, t) -> PiecewiseConstant:
        """The solution at time ``t`` as a piecewise constant function.

        On a periodic domain this is the solution on one period, in the
        representation of :class:`~fronttrack.domain.Periodic`.
        """
        t = as_fraction(t)
        return self.domain.assemble(self.active_fronts(t), t, self.initial_data)

    def evaluate(self, x, t) -> Fraction:
        return self.state_at(t)(x)

    def total_variation(self, t) -> Fraction:
        return self.state_at(t).total_variation()

    def mass(self, t, a, b) -> Fraction:
        """The integral of the solution over ``[a, b]`` at time ``t``."""
        return self.state_at(t).mass(a, b)

    def segments(self) -> list[tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]]:
        """Every front as segments in the x-t plane, cut to the domain."""
        pieces = []
        for front in self.fronts:
            pieces += self.domain.split_segment(front.endpoints(self.t_final))
        return pieces
