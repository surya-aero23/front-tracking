from fractions import Fraction

import pytest

from fronttrack.flux import (
    Burgers,
    BuckleyLeverett,
    ConcaveBurgers,
    PolygonalFlux,
    uniform_nodes,
)
from fronttrack.rational import as_fraction


def test_burgers_values_are_exact():
    flux = Burgers()
    assert flux.f(Fraction(1, 3)) == Fraction(1, 18)
    assert flux.df(Fraction(1, 3)) == Fraction(1, 3)


def test_float_input_is_read_as_a_decimal():
    assert as_fraction(0.1) == Fraction(1, 10)
    assert Burgers().f(0.5) == Fraction(1, 8)


def test_concave_burgers_is_the_negated_flux():
    assert ConcaveBurgers().f(2) == -Burgers().f(2)


def test_buckley_leverett_is_nonconvex():
    flux = BuckleyLeverett()
    nodes = uniform_nodes(0, 1, Fraction(1, 8))
    second_differences = [
        flux.f(a) - 2 * flux.f(b) + flux.f(c)
        for a, b, c in zip(nodes, nodes[1:], nodes[2:])
    ]
    assert max(second_differences) > 0
    assert min(second_differences) < 0


def test_uniform_nodes_are_anchored_at_multiples_of_delta():
    nodes = uniform_nodes(Fraction(-1, 3), Fraction(7, 8), Fraction(1, 2))
    assert nodes[0] <= Fraction(-1, 3) and nodes[-1] >= Fraction(7, 8)
    assert all(node % Fraction(1, 2) == 0 for node in nodes)


def test_refining_delta_extends_the_node_set():
    coarse = set(uniform_nodes(-1, 1, Fraction(1, 2)))
    fine = set(uniform_nodes(-1, 1, Fraction(1, 8)))
    assert coarse <= fine


def test_extra_states_become_nodes():
    nodes = uniform_nodes(0, 1, Fraction(1, 2), extra=[Fraction(1, 3)])
    assert Fraction(1, 3) in nodes


def test_polygonal_flux_interpolates_between_nodes(burgers_flux):
    # Midway between the nodes 0 and 1/4 the interpolant lies above u^2/2.
    midpoint = Fraction(1, 8)
    assert burgers_flux.f(midpoint) == Fraction(1, 64)
    assert burgers_flux.f(midpoint) > Burgers().f(midpoint)


def test_polygonal_flux_is_exact_at_nodes(burgers_flux):
    for node in burgers_flux.nodes:
        assert burgers_flux.f(node) == Burgers().f(node)


def test_polygonal_flux_rejects_states_outside_its_range(burgers_flux):
    with pytest.raises(ValueError):
        burgers_flux.f(3)


def test_nodes_in_is_a_closed_interval(burgers_flux):
    assert burgers_flux.nodes_in(0, Fraction(1, 2)) == (
        Fraction(0),
        Fraction(1, 4),
        Fraction(1, 2),
    )


def test_polygonal_flux_needs_two_nodes():
    with pytest.raises(ValueError):
        PolygonalFlux(Burgers(), [0])
