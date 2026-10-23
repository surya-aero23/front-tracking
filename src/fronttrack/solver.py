"""The front-tracking algorithm.

The method is the one of Holden and Risebro.  The flux is replaced by its
piecewise linear interpolant on a grid of mesh ``delta``, so that every
Riemann problem resolves into finitely many discontinuities and the solution
stays piecewise constant for all time.  Each front then travels in a straight
line until it meets a neighbour, and every collision is a new Riemann problem
whose solution replaces the colliding fronts.

Because all arithmetic is exact, collision times are found without tolerances
and simultaneous collisions -- including collisions of three or more fronts at
one point -- are detected as equalities.
"""

from __future__ import annotations

import heapq
from fractions import Fraction

from .domain import Domain, RealLine
from .flux import Flux, PolygonalFlux, uniform_nodes
from .fronts import Front
from .initial_data import PiecewiseConstant
from .rational import as_fraction
from .riemann import solve_riemann
from .solution import Interaction, Solution


class FrontTracker:
    """Front tracking for ``u_t + f(u)_x = 0`` with piecewise constant data."""

    def __init__(self, flux: Flux, delta, domain: Domain | None = None):
        self.base_flux = flux
        self.delta = as_fraction(delta)
        if self.delta <= 0:
            raise ValueError("delta must be positive")
        self.domain = domain if domain is not None else RealLine()

    def polygonal_flux(self, data: PiecewiseConstant) -> PolygonalFlux:
        """The flux ``f_delta`` used for this initial data.

        The nodes are the multiples of ``delta`` spanning the range of the
        data, together with the data values themselves.  Every state that ever
        appears is then a node, so the data is represented exactly and the
        node set is invariant under the computation.
        """
        nodes = uniform_nodes(data.u_min, data.u_max, self.delta, extra=data.values)
        return PolygonalFlux(self.base_flux, nodes)

    def solve(self, data: PiecewiseConstant, t_final) -> Solution:
        """Track the fronts of the solution up to time ``t_final``."""
        t_final = as_fraction(t_final)
        if t_final < 0:
            raise ValueError("t_final must be nonnegative")
        flux = self.polygonal_flux(data)
        run = _Run(flux, self.domain, data, t_final)
        run.seed_initial_fronts()
        run.advance()
        return run.finish()


class _Run:
    """The mutable state of one call to :meth:`FrontTracker.solve`.

    Fronts are held in a doubly linked list in spatial order, and candidate
    collisions in a priority queue.  Queue entries are never removed: an entry
    is simply ignored when it refers to a dead front or to a pair that is no
    longer adjacent.
    """

    def __init__(self, flux: PolygonalFlux, domain: Domain,
                 data: PiecewiseConstant, t_final: Fraction):
        self.flux = flux
        self.domain = domain
        self.data = data
        self.t_final = t_final
        self.all_fronts: list[Front] = []
        self.interactions: list[Interaction] = []
        self.head: Front | None = None
        self.tail: Front | None = None
        self.queue: list[tuple[Fraction, int, Front, Front]] = []
        self._counter = 0

    # -- construction ----------------------------------------------------

    def _new_front(self, x0, t0, speed, uL, uR) -> Front:
        front = Front(
            identifier=len(self.all_fronts),
            x0=x0, t0=t0, speed=speed, uL=uL, uR=uR,
        )
        self.all_fronts.append(front)
        return front

    def seed_initial_fronts(self) -> None:
        """Solve a Riemann problem at every breakpoint of the initial data."""
        chain: list[Front] = []
        for position, uL, uR in zip(
            self.data.breakpoints, self.data.values, self.data.values[1:]
        ):
            for wave in solve_riemann(uL, uR, self.flux):
                chain.append(
                    self._new_front(position, Fraction(0), wave.speed, wave.uL, wave.uR)
                )
        self._link(chain)
        self.head = chain[0] if chain else None
        self.tail = chain[-1] if chain else None
        for left, right in zip(chain, chain[1:]):
            self._schedule(left, right, Fraction(0))

    @staticmethod
    def _link(chain: list[Front]) -> None:
        for left, right in zip(chain, chain[1:]):
            left.next = right
            right.previous = left

    # -- event queue -----------------------------------------------------

    def _schedule(self, left: Front, right: Front, t_now) -> None:
        time = self.domain.collision_time(left, right, t_now)
        if time is None or time > self.t_final:
            return
        self._counter += 1
        heapq.heappush(self.queue, (time, self._counter, left, right))

    def advance(self) -> None:
        """Process collisions in time order until ``t_final`` is reached."""
        while self.queue:
            time, _, left, right = heapq.heappop(self.queue)
            if not (left.alive and right.alive and left.next is right):
                continue
            self._resolve(time, left, right)

    # -- interactions ----------------------------------------------------

    def _collision_group(self, time, left: Front, right: Front) -> list[Front]:
        """The maximal run of neighbouring fronts sharing the collision point.

        Exact arithmetic makes this a plain equality test, so a collision of
        three or more fronts is handled in one step rather than as a sequence
        of nearly simultaneous pair collisions.
        """
        position = self.domain.position(left, time)
        group = [left, right]
        while group[0].previous is not None and \
                self.domain.position(group[0].previous, time) == position:
            group.insert(0, group[0].previous)
        while group[-1].next is not None and \
                self.domain.position(group[-1].next, time) == position:
            group.append(group[-1].next)
        return group

    def _resolve(self, time, left: Front, right: Front) -> None:
        group = self._collision_group(time, left, right)
        position = self.domain.position(left, time)
        before, after = group[0].previous, group[-1].next

        for front in group:
            front.alive = False
            front.t_end = time

        replacements = [
            self._new_front(position, time, wave.speed, wave.uL, wave.uR)
            for wave in solve_riemann(group[0].uL, group[-1].uR, self.flux)
        ]
        self._splice(before, replacements, after)
        self.interactions.append(
            Interaction(
                time=time,
                position=position,
                incoming=tuple(front.identifier for front in group),
                outgoing=tuple(front.identifier for front in replacements),
            )
        )

        neighbours = ([before] if before else []) + replacements + ([after] if after else [])
        for a, b in zip(neighbours, neighbours[1:]):
            self._schedule(a, b, time)

    def _splice(self, before: Front | None, middle: list[Front],
                after: Front | None) -> None:
        """Replace the dead group between ``before`` and ``after`` by ``middle``."""
        self._link(middle)
        first = middle[0] if middle else after
        last = middle[-1] if middle else before

        if before is None:
            self.head = first
        else:
            before.next = first
        if first is not None:
            first.previous = before

        if after is None:
            self.tail = last
        else:
            after.previous = last
        if last is not None:
            last.next = after

    # -- result ----------------------------------------------------------

    def finish(self) -> Solution:
        for front in self.all_fronts:
            if front.t_end is None:
                front.t_end = self.t_final
        return Solution(
            flux=self.flux,
            initial_data=self.data,
            fronts=tuple(self.all_fronts),
            interactions=tuple(self.interactions),
            t_final=self.t_final,
        )
