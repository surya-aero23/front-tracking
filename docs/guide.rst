User guide
==========

This page walks through the main objects: initial data, the solver, the
solution it returns, domains, and the diagnostics.

Initial data
------------

Piecewise constant data is a :class:`~fronttrack.PiecewiseConstant`: strictly
increasing breakpoints, and one more value than there are breakpoints.  Values
and positions are stored as exact fractions; integers, fractions, strings
such as ``"1/3"`` and floats (read through their decimal representation) are
all accepted.

>>> from fronttrack import PiecewiseConstant, riemann_data
>>> data = PiecewiseConstant(breakpoints=(0, 1), values=(0, 1, 0))
>>> data(0.5), data.total_variation(), data.mass(-1, 2)
(Fraction(1, 1), Fraction(2, 1), Fraction(1, 1))
>>> riemann_data(1, 0)
PiecewiseConstant(breakpoints=(Fraction(0, 1),), values=(Fraction(1, 1), Fraction(0, 1)))

At a breakpoint the function takes its right-hand value.

Solving
-------

:class:`~fronttrack.FrontTracker` takes a flux and the mesh :math:`\delta` of
the polygonal approximation, and returns a :class:`~fronttrack.Solution`:

>>> from fronttrack import Burgers, FrontTracker
>>> solution = FrontTracker(Burgers(), delta="1/8").solve(data, t_final=4)
>>> solution.n_fronts, solution.n_interactions
(11, 2)
>>> solution.evaluate(1, 1)
Fraction(1, 1)

The nodes of :math:`f_\delta` are the multiples of :math:`\delta` spanning the
range of the data, together with the data values themselves; every state that
ever appears is a node, so the data is represented exactly.

The solution is piecewise constant at every time:

>>> state = solution.state_at(4)
>>> state.breakpoints[-1], state.values[-2:]
(Fraction(17, 6), (Fraction(3, 4), Fraction(0, 1)))

Every front is kept, with its lifetime, and every collision is recorded:

>>> event = solution.interactions[0]
>>> event.time, event.position, event.incoming, event.outgoing
(Fraction(16, 7), Fraction(15, 7), (7, 8), (9,))

Fluxes
------

Four fluxes are provided, and any subclass of :class:`~fronttrack.Flux` that
implements :meth:`~fronttrack.Flux.f` in exact arithmetic works:

==========================================  ======================================
:class:`~fronttrack.Burgers`                :math:`u^2/2`, convex
:class:`~fronttrack.ConcaveBurgers`         :math:`-u^2/2`, concave
:class:`~fronttrack.BuckleyLeverett`        :math:`u^2/(u^2 + M(1-u)^2)`, nonconvex
:class:`~fronttrack.PolygonalFlux`          the interpolant :math:`f_\delta` itself
==========================================  ======================================

A flux that sets ``convexity = 1`` (or ``-1``) lets the Riemann solver skip
the envelope where its answer is known in advance; the finite-volume schemes
use the optional vectorised ``f_array``, ``df_array`` and the list of
``extrema``.

.. code-block:: python

   from fractions import Fraction
   from fronttrack import Flux

   class Cubic(Flux):
       name = "cubic"
       extrema = ()

       def f(self, u):
           u = Fraction(u)
           return u ** 3 / 3

Periodic domains
----------------

Pass a :class:`~fronttrack.Periodic` domain to work on ``[start, start +
length)`` with the ends identified.  The data's breakpoints must lie strictly
inside the period; if ``values[0]`` and ``values[-1]`` differ, there is a jump
at the seam.

>>> from fronttrack import Periodic
>>> ring = FrontTracker(Burgers(), delta="1/4", domain=Periodic(0, 2))
>>> periodic = ring.solve(PiecewiseConstant((1,), (1, 0)), t_final=10)
>>> periodic.state_at(10).values
(Fraction(1, 2),)

Everything cancels in the end, and conservation leaves the mean.

General initial data
--------------------

:func:`~fronttrack.project` averages a function over ``cells`` cells and rounds
each average to the :math:`\delta` grid:

>>> import numpy as np
>>> from fronttrack import project
>>> sine = project(lambda x: np.sin(2 * np.pi * x), window=(0, 1), cells=32, delta="1/16")
>>> len(sine.breakpoints), sine.u_max
(29, Fraction(1, 1))

On the real line the function is taken constant outside the window.

Checking a solution
-------------------

:mod:`fronttrack.diagnostics` provides the verification used throughout the
tests:

>>> from fronttrack import check_all_fronts, mass_defect, polynomial_bump, weak_form_residual
>>> all(report.ok for report in check_all_fronts(solution))
True
>>> mass_defect(solution, t=4, a=-10, b=10)
Fraction(0, 1)
>>> abs(weak_form_residual(solution, polynomial_bump(x_center=1, x_radius=4, t_radius=3))) < 1e-12
True

and :func:`~fronttrack.l1_error`, :func:`~fronttrack.l1_distance` and
:func:`~fronttrack.run_statistics` for convergence and cost studies.

Plotting
--------

:func:`~fronttrack.plot_fronts` draws the fronts in the :math:`x`-:math:`t`
plane, shocks in red and rarefaction fronts in blue, and
:func:`~fronttrack.plot_snapshot` draws :math:`u(\cdot, t)`, optionally against
a reference.  Both take a Matplotlib axis.  See :doc:`examples`.
