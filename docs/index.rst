fronttrack
==========

**Front tracking for one-dimensional scalar conservation laws**

.. math::

   u_t + f(u)_x = 0, \qquad u(x, 0) = u_0(x).

``fronttrack`` computes weak entropy solutions by the front-tracking method of
Holden and Risebro.  The flux is replaced by its piecewise linear interpolant
:math:`f_\delta`, so that every Riemann problem resolves into finitely many
discontinuities; the solution then stays piecewise constant, every front moves
in a straight line, and each collision of fronts is a new Riemann problem.

.. plot::
   :include-source: False
   :caption: Box data for Burgers' equation: a fan (blue) overtaken by a shock (red).

   from fronttrack import Burgers, FrontTracker, PiecewiseConstant, plot_fronts, plot_snapshot

   data = PiecewiseConstant(breakpoints=(0, 1), values=(0, 1, 0))
   solution = FrontTracker(Burgers(), delta="1/8").solve(data, t_final=8)
   fig, (left, right) = plt.subplots(1, 2)
   plot_fronts(solution, left, xlim=(-0.5, 5))
   for t in (0, 1, 4, 8):
       plot_snapshot(solution, t, right, xlim=(-0.5, 5), label=f"t = {t}")
   right.legend()
   fig.tight_layout()

Highlights
----------

* **Exact.** All arithmetic is rational (:class:`fractions.Fraction`):
  collision times need no tolerance, a triple collision is one event, and
  mass, total variation and the entropy inequalities are checked exactly.
* **Any flux.** Riemann problems are solved by the convex/concave envelope
  construction, so Burgers, concave and nonconvex fluxes such as
  Buckley--Leverett all work.
* **Real line or periodic.** One event loop, two domains.
* **Any data.** Piecewise constant data directly; smooth or general data
  through cell averaging and rounding to the :math:`\delta` grid.
* **Verified.** Rankine--Hugoniot, Oleinik and Kružkov on every front; the
  weak form to round-off; property-based tests of :math:`L^1` contraction,
  comparison, maximum principle, finite speed and error bounds; exact
  references, including Buckley--Leverett and shock formation from a sine
  wave; and an independent Godunov finite-volume comparison.

.. code-block:: python

   from fronttrack import Burgers, FrontTracker, riemann_data

   solution = FrontTracker(Burgers(), delta="1/4").solve(riemann_data(1, 0), t_final=2)
   solution.state_at(2)      # PiecewiseConstant(breakpoints=(1,), values=(1, 0))

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   guide
   theory
   examples
   verification
   api
   changelog
