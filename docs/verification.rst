Verification
============

The test suite (``pytest``) checks the solver at three levels.

Every front, every run
----------------------

:func:`~fronttrack.check_all_fronts` evaluates, exactly, the Rankine--Hugoniot
condition, Oleinik's chord condition and Kružkov's jump inequality for each
front.  :func:`~fronttrack.weak_form_residual` evaluates the weak formulation
against polynomial test functions with quadrature split at the fronts and the
interaction times, so that it is exact to round-off.
:func:`~fronttrack.mass_defect` checks conservation exactly.

The theorems, on random data
----------------------------

Property-based tests (Hypothesis) draw random piecewise constant data and
check, for the computed solutions:

* :math:`L^1` contraction: :math:`\|u(t) - v(t)\|_{L^1}` is non-increasing;
* the comparison principle: :math:`u_0 \le v_0 \Rightarrow u \le v`;
* the maximum principle;
* finite speed of propagation;
* the discrete Oleinik condition: upward jumps at :math:`t > 0` are at most
  :math:`\delta` (Burgers);
* for a convex flux, each collision leaves at most one front;
* the error bound :math:`t\,\tfrac{\delta}{2}\,\operatorname{TV}(u_0)`;
* on a periodic domain, agreement with the real-line solution of the
  unrolled data, exactly;
* conservation and decay of total variation, on the line and on the ring.

These tests found a real bug -- a collision exactly at the final time was
counted twice -- that no hand-written test had hit.

Exact references
----------------

:func:`~fronttrack.exact_riemann_solution` (closed form for the quadratic
fluxes, numerical envelope with machine-precision tangency points otherwise),
:func:`~fronttrack.exact_box_solution` and
:func:`~fronttrack.exact_sine_solution` give the convergence tests their
references.  Front tracking converges at first order in all of them, and a
tracked shock has no error at all.

An independent scheme
---------------------

:class:`~fronttrack.GodunovSolver` shares nothing with the front tracker but
the flux; its solutions converge to the front-tracking ones, including for
Buckley--Leverett.
