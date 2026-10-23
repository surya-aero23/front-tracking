"""Piecewise constant initial data.

A piecewise constant function is stored as a strictly increasing list of
breakpoints together with the constant values of the cells, one more value
than there are breakpoints.  The Riemann problem is the special case of a
single breakpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .rational import as_fraction


@dataclass(frozen=True)
class PiecewiseConstant:
    """``u(x) = values[i]`` for ``breakpoints[i-1] < x < breakpoints[i]``."""

    breakpoints: tuple[Fraction, ...]
    values: tuple[Fraction, ...]

    def __init__(self, breakpoints, values):
        breakpoints = tuple(as_fraction(x) for x in breakpoints)
        values = tuple(as_fraction(v) for v in values)
        if len(values) != len(breakpoints) + 1:
            raise ValueError("expected one value more than there are breakpoints")
        if any(a >= b for a, b in zip(breakpoints, breakpoints[1:])):
            raise ValueError("breakpoints must be strictly increasing")
        object.__setattr__(self, "breakpoints", breakpoints)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_jumps(cls, u_left, jumps) -> "PiecewiseConstant":
        """Build from a left state and ``(position, new value)`` pairs.

        Pairs at a common position are collapsed, keeping the rightmost
        value, and jumps between equal values are dropped.  This is how the
        solver turns a list of fronts back into a function.
        """
        u_left = as_fraction(u_left)
        collapsed: list[tuple[Fraction, Fraction]] = []
        for position, value in jumps:
            position, value = as_fraction(position), as_fraction(value)
            if collapsed and collapsed[-1][0] == position:
                collapsed[-1] = (position, value)
            else:
                collapsed.append((position, value))

        breakpoints: list[Fraction] = []
        values: list[Fraction] = [u_left]
        for position, value in collapsed:
            if value == values[-1]:
                continue
            breakpoints.append(position)
            values.append(value)
        return cls(tuple(breakpoints), tuple(values))

    def __call__(self, x) -> Fraction:
        """The value at ``x``; at a breakpoint the right-hand value is taken."""
        x = as_fraction(x)
        index = 0
        while index < len(self.breakpoints) and x >= self.breakpoints[index]:
            index += 1
        return self.values[index]

    @property
    def u_left(self) -> Fraction:
        return self.values[0]

    @property
    def u_right(self) -> Fraction:
        return self.values[-1]

    @property
    def u_min(self) -> Fraction:
        return min(self.values)

    @property
    def u_max(self) -> Fraction:
        return max(self.values)

    def total_variation(self) -> Fraction:
        return sum(
            (abs(b - a) for a, b in zip(self.values, self.values[1:])),
            Fraction(0),
        )

    def mass(self, a, b) -> Fraction:
        """The integral of ``u`` over ``[a, b]``."""
        a, b = as_fraction(a), as_fraction(b)
        if a > b:
            raise ValueError("expected a <= b")
        edges = (a,) + self.breakpoints + (b,)
        total = Fraction(0)
        for index, value in enumerate(self.values):
            left = max(a, edges[index])
            right = min(b, edges[index + 1])
            if right > left:
                total += value * (right - left)
        return total

    def cells(self):
        """Yield ``(left edge, right edge, value)``, with ``None`` at infinity."""
        edges = (None,) + self.breakpoints + (None,)
        for index, value in enumerate(self.values):
            yield edges[index], edges[index + 1], value


def riemann_data(uL, uR, x0=0) -> PiecewiseConstant:
    """The Riemann problem: ``uL`` for ``x < x0`` and ``uR`` for ``x > x0``."""
    return PiecewiseConstant((x0,), (uL, uR))
