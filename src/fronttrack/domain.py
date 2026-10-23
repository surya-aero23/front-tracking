"""The spatial domain.

The domain owns the geometry of the front configuration: where the initial
jumps sit, how far apart two neighbouring fronts are, how positions are
reported, and how the fronts alive at one time are turned back into a
function.  The solver itself never looks at coordinates directly, so the real
line and the periodic interval share one event loop.

On a periodic domain the fronts form a ring.  Each front keeps a *lifted*
position on the real line, so that it moves in a straight line forever; the
domain reduces positions to the period only when they are reported.  Along the
ring the lifted positions increase, except across the seam between the last
front (the ring's tail) and the first (its head), where one period is added.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction

from .fronts import Front
from .initial_data import PiecewiseConstant
from .rational import as_fraction

Segment = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]


class Domain(ABC):
    """Interface between the solver and the geometry of the line."""

    name: str = "domain"

    @abstractmethod
    def initial_jumps(self, data: PiecewiseConstant) -> list[tuple[Fraction, Fraction, Fraction]]:
        """The jumps ``(position, uL, uR)`` of the data, in spatial order."""

    @abstractmethod
    def gap(self, left: Front, right: Front, t, across_seam=False) -> Fraction:
        """The distance from ``left`` to its right neighbour ``right``."""

    @abstractmethod
    def position(self, front: Front, t) -> Fraction:
        """The position of ``front`` at time ``t``, reduced to the domain."""

    @abstractmethod
    def assemble(self, fronts: list[Front], t, data: PiecewiseConstant) -> PiecewiseConstant:
        """The solution at ``t`` from its fronts, sorted by reduced position."""

    def wraps_around(self) -> bool:
        """Whether the last and first fronts are also neighbours."""
        return False

    def collision_time(self, left: Front, right: Front, t_now,
                       across_seam=False) -> Fraction | None:
        """When the neighbours ``left`` and ``right`` meet after ``t_now``."""
        t_now = as_fraction(t_now)
        closing = left.speed - right.speed
        if closing <= 0:
            return None
        gap = self.gap(left, right, t_now, across_seam)
        if gap < 0:
            raise ValueError("fronts are out of order")
        meeting = t_now + gap / closing
        return meeting if meeting > t_now else None

    def split_segment(self, segment: Segment) -> list[Segment]:
        """A front's x-t segment, cut into pieces that lie in the domain."""
        return [segment]


class RealLine(Domain):
    """The whole real line, with constant states at both infinities."""

    name = "real-line"

    def initial_jumps(self, data):
        return list(zip(data.breakpoints, data.values, data.values[1:]))

    def gap(self, left, right, t, across_seam=False):
        return right.position(t) - left.position(t)

    def position(self, front, t):
        return front.position(t)

    def assemble(self, fronts, t, data):
        jumps = [(front.position(t), front.uR) for front in fronts]
        return PiecewiseConstant.from_jumps(data.u_left, jumps)


class Periodic(Domain):
    """The interval ``[start, start + length)`` with its ends identified.

    Data on a periodic domain is a :class:`PiecewiseConstant` whose
    breakpoints lie strictly inside the period; ``values[0]`` is the state
    just right of ``start`` and ``values[-1]`` the state just left of
    ``start + length``.  If the two differ there is a jump at the seam.
    """

    name = "periodic"

    def __init__(self, start=0, length=1):
        self.start = as_fraction(start)
        self.length = as_fraction(length)
        if self.length <= 0:
            raise ValueError("the period must be positive")

    @property
    def end(self) -> Fraction:
        return self.start + self.length

    def wraps_around(self) -> bool:
        return True

    def reduce(self, x) -> Fraction:
        """``x`` moved into ``[start, start + length)``."""
        return self.start + (as_fraction(x) - self.start) % self.length

    def initial_jumps(self, data):
        if any(not self.start < x < self.end for x in data.breakpoints):
            raise ValueError("breakpoints must lie strictly inside the period")
        jumps = []
        if data.values[-1] != data.values[0]:
            jumps.append((self.start, data.values[-1], data.values[0]))
        jumps += list(zip(data.breakpoints, data.values, data.values[1:]))
        return jumps

    def gap(self, left, right, t, across_seam=False):
        gap = right.position(t) - left.position(t)
        return gap + self.length if across_seam else gap

    def position(self, front, t):
        return self.reduce(front.position(t))

    def assemble(self, fronts, t, data):
        if not fronts:
            # Every front has cancelled; mass conservation fixes the constant.
            return PiecewiseConstant((), (self.mean(data),))
        jumps = [(self.position(front, t), front.uR) for front in fronts]
        state = PiecewiseConstant.from_jumps(fronts[-1].uR, jumps)
        if state.breakpoints and state.breakpoints[0] == self.start:
            # A jump sitting on the seam is the jump into values[0]; keep the
            # representation with breakpoints strictly inside the period.
            state = PiecewiseConstant(state.breakpoints[1:], state.values[1:])
        return state

    def mean(self, data: PiecewiseConstant) -> Fraction:
        """The mean of the data over one period."""
        return data.mass(self.start, self.end) / self.length

    def split_segment(self, segment):
        (x0, t0), (x1, t1) = segment
        pieces = []
        # Shift the segment so that it starts inside the period, then cut it
        # at every multiple of the period it crosses.
        shift = self.reduce(x0) - x0
        x0, x1 = x0 + shift, x1 + shift
        while True:
            if self.start <= x1 < self.end or x1 == x0:
                pieces.append(((x0, t0), (x1, t1)))
                return pieces
            edge = self.end if x1 >= self.end else self.start
            t_cut = t0 + (t1 - t0) * (edge - x0) / (x1 - x0)
            pieces.append(((x0, t0), (edge, t_cut)))
            x0 = self.start if edge == self.end else self.end
            x1 = x1 - self.length if edge == self.end else x1 + self.length
            t0 = t_cut
