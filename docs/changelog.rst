Changelog
=========

0.2.0
-----

* Periodic domains (:class:`~fronttrack.Periodic`).
* General initial data through :func:`~fronttrack.project`, and the sine-wave
  shock-formation study with :func:`~fronttrack.exact_sine_solution`.
* Exact Riemann solutions for any smooth flux, by the envelope construction
  on the true flux; the Buckley--Leverett Welge point to machine precision.
* The expansion shock (:func:`~fronttrack.expansion_shock`) and the
  Murman--Roe scheme (:class:`~fronttrack.RoeSolver`).
* Property-based tests of the theory, and :func:`~fronttrack.l1_distance`.
* :func:`~fronttrack.run_statistics`, and a convexity shortcut in the Riemann
  solver that makes Burgers runs linear in :math:`1/\delta`.
* Fixed: a collision exactly at the final time was counted twice in
  :meth:`~fronttrack.Solution.state_at`.

0.1.0
-----

* Front tracking on the real line with exact rational arithmetic.
* Envelope-based Riemann solver for any flux; Burgers, concave Burgers and
  Buckley--Leverett.
* Rankine--Hugoniot, Oleinik and Kružkov checks; weak-form residual; mass
  conservation; :math:`L^1` errors against exact solutions.
* Godunov finite-volume scheme for comparison.
* Plots of fronts and snapshots.
