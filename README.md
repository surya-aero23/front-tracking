# front-tracking-burgers

A front-tracking solver for the 1D scalar conservation law

    u_t + f(u)_x = 0,    u(x, 0) = u_0(x)

with piecewise constant (in particular Riemann) initial data on the real line.
The default flux is Burgers' flux `f(u) = u^2 / 2`.

Course project for MA 347B, *Hyperbolic Conservation Laws: Theory, Numerics and
Applications*.

Work in progress; see `report/main.tex` for the write-up.
