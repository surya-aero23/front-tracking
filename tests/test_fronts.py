from fractions import Fraction

import pytest

from fronttrack.domain import RealLine
from fronttrack.fronts import Front


def make_front(x0=0, t0=0, speed=1, uL=1, uR=0, identifier=0):
    return Front(identifier=identifier, x0=x0, t0=t0, speed=speed, uL=uL, uR=uR)


def test_position_is_linear_in_time():
    front = make_front(x0=1, t0=Fraction(1, 2), speed=2)
    assert front.position(Fraction(3, 2)) == 3


def test_a_front_needs_two_distinct_states():
    with pytest.raises(ValueError):
        make_front(uL=1, uR=1)


def test_lifetime_is_half_open():
    front = make_front()
    front.t_end = Fraction(1)
    assert front.alive_at(0, 2) and front.alive_at(Fraction(1, 2), 2)
    assert not front.alive_at(1, 2)


def test_a_front_surviving_to_the_final_time_is_alive_there():
    front = make_front()
    front.t_end = Fraction(2)
    assert front.alive_at(2, 2)


def test_collision_time_of_converging_fronts():
    left = make_front(x0=0, speed=2, identifier=0)
    right = make_front(x0=1, speed=1, identifier=1)
    assert RealLine().collision_time(left, right, 0) == 1


def test_diverging_fronts_never_collide():
    left = make_front(x0=0, speed=1)
    right = make_front(x0=1, speed=2)
    assert RealLine().collision_time(left, right, 0) is None


def test_parallel_fronts_never_collide():
    left = make_front(x0=0, speed=1)
    right = make_front(x0=1, speed=1)
    assert RealLine().collision_time(left, right, 0) is None
