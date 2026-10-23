"""Convex and concave envelopes of a flux graph.

These implement the geometric construction of Lecture 8&9, section 5.13: the
entropy solution of a Riemann problem is read off the lower convex envelope of
the flux when ``u_l < u_r`` and off the upper concave envelope when
``u_l > u_r``.  Straight segments of the envelope are shocks and the vertices
are the intermediate states.

Collinear interior points are dropped, so that a straight stretch of the
envelope yields a single front rather than several fronts of equal speed.
"""

from __future__ import annotations

from fractions import Fraction

Point = tuple[Fraction, Fraction]


def _cross(o: Point, a: Point, b: Point) -> Fraction:
    """Twice the signed area of the triangle ``o, a, b``.

    Positive means ``o -> a -> b`` turns counter-clockwise, i.e. the point
    ``a`` lies below the chord from ``o`` to ``b``.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _hull(points: list[Point], turn: int) -> list[Point]:
    """Monotone chain hull keeping only turns of the requested sign."""
    hull: list[Point] = []
    for point in points:
        while len(hull) >= 2 and turn * _cross(hull[-2], hull[-1], point) <= 0:
            hull.pop()
        hull.append(point)
    return hull


def lower_convex_envelope(points) -> list[Point]:
    """Vertices of the largest convex function lying below the given graph.

    ``points`` must be ``(u, f(u))`` pairs with strictly increasing ``u``.  The
    first and last points are always vertices.
    """
    points = _validated(points)
    if len(points) <= 2:
        return points
    return _hull(points, turn=+1)


def upper_concave_envelope(points) -> list[Point]:
    """Vertices of the smallest concave function lying above the given graph."""
    points = _validated(points)
    if len(points) <= 2:
        return points
    return _hull(points, turn=-1)


def _validated(points) -> list[Point]:
    points = [(u, value) for u, value in points]
    if len(points) < 2:
        raise ValueError("an envelope needs at least two points")
    for left, right in zip(points, points[1:]):
        if left[0] >= right[0]:
            raise ValueError("points must have strictly increasing abscissae")
    return points
