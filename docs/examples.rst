Examples
========

Every figure on this page is produced by the code shown with it, when the
documentation is built.

The two Riemann problems
------------------------

.. plot::

   from fronttrack import Burgers, FrontTracker, exact_riemann_solution, plot_fronts, plot_snapshot, riemann_data

   fig, axes = plt.subplots(1, 2)
   for ax, (uL, uR) in zip(axes, [(1, 0), (0, 1)]):
       solution = FrontTracker(Burgers(), delta="1/8").solve(riemann_data(uL, uR), t_final=1)
       plot_snapshot(solution, 1, ax, xlim=(-1, 2), reference=exact_riemann_solution(uL, uR),
                     label="front tracking")
       ax.set_title(f"$u_l = {uL}$, $u_r = {uR}$")
       ax.legend()
   fig.tight_layout()

Three shocks meeting at one point
---------------------------------

Exact arithmetic makes the triple collision a single event.

.. plot::

   from fronttrack import Burgers, FrontTracker, PiecewiseConstant, plot_fronts

   data = PiecewiseConstant((Fraction(-5, 2), Fraction(-3, 2), Fraction(-1, 2)), (3, 2, 1, 0))
   solution = FrontTracker(Burgers(), delta="1/8").solve(data, t_final=2)
   (event,) = solution.interactions
   print(event)
   fig, ax = plt.subplots(figsize=(4.5, 3.4))
   plot_fronts(solution, ax, xlim=(-3, 3))
   fig.tight_layout()

Buckley--Leverett
-----------------

A nonconvex flux: a rarefaction running downwards, then a shock from the
Welge point :math:`u^* = 1/\sqrt{3}`.

.. plot::

   from fronttrack import BuckleyLeverett, FrontTracker, exact_riemann_solution, plot_fronts, plot_snapshot, riemann_data

   flux = BuckleyLeverett()
   solution = FrontTracker(flux, delta="1/32").solve(riemann_data(1, 0), t_final=1)
   fig, (left, right) = plt.subplots(1, 2)
   plot_fronts(solution, left, xlim=(-0.2, 2))
   plot_snapshot(solution, 1, right, xlim=(-0.2, 2),
                 reference=exact_riemann_solution(1, 0, flux), label="front tracking")
   right.legend()
   fig.tight_layout()

Shock formation on a period
---------------------------

Smooth periodic data, projected onto cells; the shock forms at
:math:`t_b = 1/2\pi`.

.. plot::

   from fronttrack import (Burgers, FrontTracker, Periodic, exact_sine_solution,
                           plot_fronts, plot_snapshot, project)

   delta = Fraction(1, 16)
   data = project(lambda x: np.sin(2 * np.pi * x), (0, 1), cells=32, delta=delta)
   solution = FrontTracker(Burgers(), delta, Periodic(0, 1)).solve(data, Fraction(1, 2))
   exact = exact_sine_solution()
   fig, (left, right) = plt.subplots(1, 2)
   plot_fronts(solution, left, xlim=(0, 1))
   left.axhline(exact.breaking_time, color="0.4", linestyle=":")
   plot_snapshot(solution, Fraction(3, 10), right, xlim=(0, 1), reference=exact,
                 label="front tracking, t = 0.3")
   right.legend()
   fig.tight_layout()

Two weak solutions, and a scheme that picks the wrong one
---------------------------------------------------------

The expansion shock satisfies Rankine--Hugoniot and the weak form, but not the
entropy condition.  The Murman--Roe scheme converges to it on a transonic
rarefaction; Godunov's scheme opens the fan.

.. plot::

   from fronttrack import (Burgers, GodunovSolver, RoeSolver, check_all_fronts,
                           exact_riemann_solution, expansion_shock, riemann_data)

   (report,) = check_all_fronts(expansion_shock(0, 1, t_final=1))
   print(report)

   data, window = riemann_data(-1, 1), (-2, 2)
   fig, ax = plt.subplots(figsize=(5, 3.4))
   for scheme in (GodunovSolver, RoeSolver):
       result = scheme(Burgers(), cells=200, window=window).solve(data, t_final=1)
       ax.step(result.edges[:-1], result.averages, where="post", label=scheme.__name__)
   x = np.linspace(-2, 2, 400)
   exact = exact_riemann_solution(-1, 1)
   ax.plot(x, [float(exact(Fraction(str(v)), 1)) for v in x], "k--", lw=0.8, label="exact")
   ax.legend()
   fig.tight_layout()

Front tracking against finite volumes
-------------------------------------

.. plot::

   import time
   from fronttrack import (Burgers, FrontTracker, GodunovSolver, PiecewiseConstant,
                           exact_box_solution, l1_error)

   data, exact, t = PiecewiseConstant((0, 1), (0, 1, 0)), exact_box_solution(), 3
   fig, ax = plt.subplots(figsize=(5, 3.4))
   tracked, godunov = [], []
   for k in range(2, 8):
       start = time.perf_counter()
       solution = FrontTracker(Burgers(), Fraction(1, 2 ** k)).solve(data, t)
       tracked.append((time.perf_counter() - start,
                       l1_error(solution, exact, t, (-1, 4), extra_breaks=[0, 6 ** 0.5])))
       start = time.perf_counter()
       result = GodunovSolver(Burgers(), 25 * 2 ** k, (-1, 4)).solve(data, t)
       godunov.append((time.perf_counter() - start, result.l1_error(exact)))
   ax.loglog(*zip(*tracked), "o-", label="front tracking")
   ax.loglog(*zip(*godunov), "s-.", label="Godunov")
   ax.set_xlabel("CPU time [s]")
   ax.set_ylabel("$L^1$ error at $t = 3$")
   ax.legend()
   fig.tight_layout()
