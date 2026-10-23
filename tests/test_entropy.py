from fractions import Fraction

from fronttrack.entropy import (
    is_admissible,
    kruzkov_admissible,
    oleinik_admissible,
    rankine_hugoniot_ok,
    shock_speed,
)
from fronttrack.flux import BuckleyLeverett, PolygonalFlux, uniform_nodes
from fronttrack.riemann import solve_riemann


def test_shock_speed_is_the_chord_slope(burgers_flux):
    assert shock_speed(1, -1, burgers_flux) == Fraction(0)
    assert shock_speed(2, 1, burgers_flux) == Fraction(3, 2)


def test_rankine_hugoniot_rejects_a_wrong_speed(burgers_flux):
    assert rankine_hugoniot_ok(1, -1, 0, burgers_flux)
    assert not rankine_hugoniot_ok(1, -1, Fraction(1, 10), burgers_flux)


def test_decreasing_jump_is_entropy_admissible(burgers_flux):
    speed = shock_speed(2, -2, burgers_flux)
    assert is_admissible(2, -2, speed, burgers_flux)


def test_increasing_jump_is_not_a_shock(burgers_flux):
    # Lecture 8&9, section 5.5: this weak solution is rejected by entropy.
    speed = shock_speed(-1, 1, burgers_flux)
    assert rankine_hugoniot_ok(-1, 1, speed, burgers_flux)
    assert not oleinik_admissible(-1, 1, speed, burgers_flux)
    assert not kruzkov_admissible(-1, 1, speed, burgers_flux)


def test_smallest_increasing_jump_is_admissible(burgers_flux):
    # A jump across a single cell of the polygonal flux has no state in
    # between, so it carries no entropy violation: these are the fronts that
    # make up a rarefaction fan.
    speed = shock_speed(0, Fraction(1, 4), burgers_flux)
    assert is_admissible(0, Fraction(1, 4), speed, burgers_flux)


def test_every_riemann_front_is_admissible(burgers_flux):
    states = [-2, -1, Fraction(-1, 2), 0, Fraction(3, 4), 1, 2]
    for uL in states:
        for uR in states:
            for wave in solve_riemann(uL, uR, burgers_flux):
                assert is_admissible(wave.uL, wave.uR, wave.speed, burgers_flux)


def test_every_nonconvex_riemann_front_is_admissible():
    flux = PolygonalFlux(BuckleyLeverett(), uniform_nodes(0, 1, Fraction(1, 8)))
    for uL in flux.nodes:
        for uR in flux.nodes:
            for wave in solve_riemann(uL, uR, flux):
                assert is_admissible(wave.uL, wave.uR, wave.speed, flux)


def test_nonconvex_direct_shock_can_violate_entropy():
    # Even with u_l > u_r the chord may cross the graph: Lecture 8&9, 5.12.
    flux = PolygonalFlux(BuckleyLeverett(), uniform_nodes(0, 1, Fraction(1, 8)))
    speed = shock_speed(1, 0, flux)
    assert rankine_hugoniot_ok(1, 0, speed, flux)
    assert not oleinik_admissible(1, 0, speed, flux)
