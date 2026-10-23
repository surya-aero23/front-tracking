"""The spatial domain.

The domain owns the geometry of the front configuration: which fronts are
neighbours and when two neighbours collide.  Only the real line is
implemented; a periodic domain would subclass :class:`Domain` and add the
wrap-around neighbour pair together with a position reduction, without
touching the solver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction

from .fronts import Front
from .rational import as_fraction


class Domain(ABC):
    """Interface between the solver and the geometry of the line."""

    name: str = "domain"

    @abstractmethod
    def collision_time(self, left: Front, right: Front, t_now) -> Fraction | None:
        """When the neighbours ``left`` and ``right`` meet after ``t_now``."""

    @abstractmethod
    def position(self, front: Front, t) -> Fraction:
        """The position of ``front`` at time ``t``, reduced to the domain."""

    def wraps_around(self) -> bool:
        """Whether the leftmost and rightmost fronts are also neighbours."""
        return False


class RealLine(Domain):
    """The whole real line, with constant states at both infinities."""

    name = "real-line"

    def collision_time(self, left: Front, right: Front, t_now) -> Fraction | None:
        t_now = as_fraction(t_now)
        closing = left.speed - right.speed
        if closing <= 0:
            return None
        gap = right.position(t_now) - left.position(t_now)
        if gap < 0:
            raise ValueError("fronts are out of order")
        meeting = t_now + gap / closing
        return meeting if meeting > t_now else None

    def position(self, front: Front, t) -> Fraction:
        return front.position(t)
