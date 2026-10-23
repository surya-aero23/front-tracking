from fractions import Fraction

import pytest

from fronttrack.flux import (
    BuckleyLeverett,
    Burgers,
    ConcaveBurgers,
    PolygonalFlux,
    uniform_nodes,
)
from fronttrack.riemann import solve_riemann


def test_equal_states_give_no_waves(burgers_flux):
    assert solve_riemann(1, 1, burgers_flux) == []


def test_decreasing_jump_is_a_single_shock(burgers_flux):
    # Lecture 8&9, section 5.8.1: u_l > u_r gives one shock at (u_l + u_r)/2.
    waves = solve_riemann(1, -1, burgers_flux)
    assert len(waves) == 1
    assert waves[0].speed == Fraction(0)

    waves = solve_riemann(2, Fraction(1, 2), burgers_flux)
    assert len(waves) == 1
    assert waves[0].speed == Fraction(5, 4)


def test_increasing_jump_is_a_fan_of_one_front_per_cell(burgers_flux):
    waves = solve_riemann(0, 1, burgers_flux)
    assert len(waves) == 4
    # Each front connects neighbouring nodes and moves at the midpoint speed.
    for wave in waves:
        assert wave.uR - wave.uL == Fraction(1, 4)
        assert wave.speed == (wave.uL + wave.uR) / 2


def test_fan_speeds_are_strictly_increasing(burgers_flux):
    for uL, uR in [(0, 2), (-2, 2), (Fraction(-1, 2), Fraction(3, 4))]:
        speeds = [wave.speed for wave in solve_riemann(uL, uR, burgers_flux)]
        assert all(a < b for a, b in zip(speeds, speeds[1:]))


def test_waves_chain_the_two_states_together(burgers_flux):
    for uL, uR in [(0, 2), (2, -1), (-2, Fraction(1, 2))]:
        waves = solve_riemann(uL, uR, burgers_flux)
        assert waves[0].uL == uL
        assert waves[-1].uR == uR
        assert all(a.uR == b.uL for a, b in zip(waves, waves[1:]))


def test_fan_refines_with_delta():
    for delta, expected in [(Fraction(1, 4), 4), (Fraction(1, 8), 8)]:
        flux = PolygonalFlux(Burgers(), uniform_nodes(0, 1, delta))
        assert len(solve_riemann(0, 1, flux)) == expected


def test_concave_flux_reverses_the_wave_rules():
    # Lecture 8&9, section 5.17: for a concave flux the roles are swapped.
    flux = PolygonalFlux(ConcaveBurgers(), uniform_nodes(0, 1, Fraction(1, 4)))
    assert len(solve_riemann(0, 1, flux)) == 1
    assert len(solve_riemann(1, 0, flux)) == 4


def test_nonconvex_flux_gives_a_composite_wave():
    # Buckley-Leverett: a rarefaction followed by a shock, Lecture 8&9 5.17.
    flux = PolygonalFlux(BuckleyLeverett(), uniform_nodes(0, 1, Fraction(1, 8)))
    waves = solve_riemann(1, 0, flux)
    assert len(waves) > 1
    jumps = [wave.uR - wave.uL for wave in waves]
    # Several small rarefaction fronts, then one large jump down to zero.
    assert waves[-1].uR == 0
    assert abs(jumps[-1]) > abs(jumps[0])


def test_states_must_be_nodes(burgers_flux):
    with pytest.raises(ValueError):
        solve_riemann(Fraction(1, 3), 1, burgers_flux)
