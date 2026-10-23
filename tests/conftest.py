from fractions import Fraction

import pytest

from fronttrack.flux import Burgers, PolygonalFlux, uniform_nodes


@pytest.fixture
def burgers_flux():
    """A polygonal Burgers flux of mesh 1/4 on the states [-2, 2]."""
    return PolygonalFlux(Burgers(), uniform_nodes(-2, 2, Fraction(1, 4)))
