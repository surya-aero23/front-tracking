Theory
======

This page summarises the mathematics the code implements.

Weak and entropy solutions
--------------------------

A bounded :math:`u` is a *weak solution* of :math:`u_t + f(u)_x = 0`,
:math:`u(\cdot, 0) = u_0`, if for every :math:`\varphi \in C^1_c(\mathbb{R}
\times [0, \infty))`

.. math::

   \int_0^\infty\!\!\int_{\mathbb{R}} \bigl(u\,\varphi_t + f(u)\,\varphi_x\bigr)\,dx\,dt
   + \int_{\mathbb{R}} u_0(x)\,\varphi(x, 0)\,dx = 0.

A piecewise constant :math:`u` whose jumps move along straight lines is a weak
solution exactly when every jump satisfies the **Rankine--Hugoniot** condition

.. math::

   s\,(u_r - u_l) = f(u_r) - f(u_l).

Weak solutions are not unique.  The *entropy solution* is singled out by
**Oleinik's chord condition** -- the graph of :math:`f` lies below the chord
from :math:`u_l` to :math:`u_r` for a decreasing jump, and above it for an
increasing one -- or equivalently by **Kružkov's** inequalities: for every
constant :math:`k`, with :math:`U_k = |u - k|` and
:math:`F_k = \operatorname{sgn}(u - k)(f(u) - f(k))`,

.. math::

   s\,[U_k] \ge [F_k] \quad\text{across every jump.}

The Riemann problem by envelopes
--------------------------------

For data :math:`u_l` for :math:`x < 0` and :math:`u_r` for :math:`x > 0`, the
entropy solution is read off a convex hull:

* if :math:`u_l < u_r`, take the **lower convex envelope** of :math:`f` on
  :math:`[u_l, u_r]`;
* if :math:`u_l > u_r`, take the **upper concave envelope** on
  :math:`[u_r, u_l]`.

Straight pieces of the envelope are shocks moving at their slope; pieces where
the envelope touches the graph are rarefaction fans, along which
:math:`f'(u) = x/t`.  For Burgers' flux this reduces to the familiar rule: a
decreasing jump is a shock at speed :math:`(u_l + u_r)/2`, an increasing one
is the fan :math:`u = x/t`.  For a nonconvex flux a single Riemann problem can
produce both.

The polygonal flux
------------------

Replace :math:`f` by its piecewise linear interpolant :math:`f_\delta` on nodes
spaced :math:`\delta` apart.  The envelope of a polygon is a polygon, so every
Riemann solution of :math:`f_\delta` consists of discontinuities only, with
intermediate states at the nodes.  Hence:

1. a fan becomes a staircase of small jumps, one per :math:`\delta`-cell;
2. no new states are ever created, so the node set is invariant;
3. the solution is piecewise constant at all times, and every front is a
   straight line in the :math:`x`-:math:`t` plane.

The solution of the polygonal problem is computed *exactly*; the only error
is :math:`f \ne f_\delta`, which by Kružkov's stability estimate gives

.. math::

   \|u_\delta(t) - u(t)\|_{L^1} \le t\,\operatorname{Lip}(f - f_\delta)\,\operatorname{TV}(u_0),

that is :math:`t\,\tfrac{\delta}{2}\,\operatorname{TV}(u_0)` for Burgers.

The algorithm
-------------

1. Solve a Riemann problem at every jump of the data and emit the fronts.
2. For each pair of neighbouring fronts that approach each other, compute when
   they meet, and keep these times in a priority queue.
3. Take the earliest collision, gather every front at that point, and replace
   them by the solution of the Riemann problem between the outer states.
   Schedule the new neighbour pairs.
4. Repeat until the final time.

For a convex flux each collision leaves at most one front behind, so the
number of fronts never exceeds its initial value and the number of
collisions is bounded by it.

Because all arithmetic is rational, simultaneous collisions -- three or more
fronts at one point, or several collisions at one time -- are recognised by
equality tests, with no tolerance.

Periodic domains
----------------

On :math:`[a, a + L)` the fronts form a ring.  Each front keeps a lifted
position on :math:`\mathbb{R}` and moves in a straight line forever; positions
are reduced modulo :math:`L` only when reported.  The last and first fronts
are neighbours across the seam, with :math:`L` added to their gap.

General data
------------

General :math:`u_0` is averaged over cells of width :math:`h` and each average
is rounded to the nearest multiple of :math:`\delta`, so that every state is a
node of :math:`f_\delta`.  The error is then :math:`O(h + \delta)` for smooth
data.

References
----------

* H. Holden and N. H. Risebro, *Front Tracking for Hyperbolic Conservation
  Laws*, Springer, 2nd ed., 2015.
* S. N. Kružkov, First order quasilinear equations in several independent
  variables, *Math. USSR Sbornik* 10 (1970), 217--243.
* O. A. Oleinik, Discontinuous solutions of non-linear differential
  equations, *AMS Translations* 26 (1963), 95--172.
