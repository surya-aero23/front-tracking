"""Exact rational arithmetic helpers.

Front tracking for a piecewise linear flux only ever needs the four
arithmetic operations: chord slopes, front positions and collision times are
all rational functions of the data.  Working in ``Fraction`` therefore makes
the algorithm exact, so that simultaneous collisions and wave cancellations
are detected by equality rather than by a tolerance.
"""

from fractions import Fraction
from numbers import Rational

Number = Fraction


def as_fraction(x) -> Fraction:
    """Coerce ``x`` to an exact ``Fraction``.

    Floats are converted through their decimal representation, so that
    ``as_fraction(0.1)`` is ``1/10`` rather than the binary value actually
    stored.  This is what a user writing ``0.1`` means.
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, Rational):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(str(x))
    if isinstance(x, str):
        return Fraction(x)
    raise TypeError(f"cannot interpret {x!r} as an exact rational number")


def sign(x) -> int:
    """The sign of a rational number, as -1, 0 or +1."""
    return (x > 0) - (x < 0)
