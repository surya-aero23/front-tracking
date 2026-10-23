from fractions import Fraction

import pytest

from fronttrack.envelope import lower_convex_envelope, upper_concave_envelope


def _points(pairs):
    return [(Fraction(u), Fraction(v)) for u, v in pairs]


def test_convex_graph_is_its_own_lower_envelope():
    points = _points([(0, 0), (1, 1), (2, 4), (3, 9)])
    assert lower_convex_envelope(points) == points


def test_lower_envelope_cuts_off_a_concave_bump():
    # The interior point lies above the chord and is therefore discarded.
    points = _points([(0, 0), (1, 1), (2, 0)])
    assert lower_convex_envelope(points) == _points([(0, 0), (2, 0)])


def test_upper_envelope_cuts_off_a_convex_dip():
    points = _points([(0, 0), (1, -1), (2, 0)])
    assert upper_concave_envelope(points) == _points([(0, 0), (2, 0)])


def test_collinear_points_are_dropped():
    points = _points([(0, 0), (1, 1), (2, 2), (3, 3)])
    assert lower_convex_envelope(points) == _points([(0, 0), (3, 3)])
    assert upper_concave_envelope(points) == _points([(0, 0), (3, 3)])


def test_envelope_endpoints_are_always_kept():
    points = _points([(0, 5), (1, -3), (2, 4)])
    for envelope in (lower_convex_envelope, upper_concave_envelope):
        vertices = envelope(points)
        assert vertices[0] == points[0]
        assert vertices[-1] == points[-1]


def test_non_monotone_abscissae_are_rejected():
    with pytest.raises(ValueError):
        lower_convex_envelope(_points([(1, 0), (0, 0)]))
