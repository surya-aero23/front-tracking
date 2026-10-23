# front-tracking-burgers

A front-tracking solver for the one-dimensional scalar conservation law

    u_t + f(u)_x = 0,    u(x, 0) = u_0(x),    x in R

with piecewise constant initial data, in particular the Riemann problem.
Burgers' flux `f(u) = u^2 / 2` is the default; any flux can be supplied.

Course project for MA 347B, *Hyperbolic Conservation Laws: Theory, Numerics
and Applications*.  The write-up is in [`report/main.tex`](report/main.tex).

## The method

No mesh in `x` or `t` is used.  The flux is replaced by its piecewise linear
interpolant `f_delta` on a grid of spacing `delta` in state space, so every
Riemann solution is made of finitely many discontinuities and the solution
stays piecewise constant for all time.  Each front then travels in a straight
line until it meets a neighbour, and every collision is resolved as a new
Riemann problem.

The Riemann solver is the geometric one: the intermediate states are the
vertices of the lower convex envelope of the flux when `u_l < u_r`, and of the
upper concave envelope when `u_l > u_r`.  Straight segments are shocks and
curved portions rarefactions.  This is correct for any flux, convex or not.

All arithmetic is exact (`fractions.Fraction`), so collision times are found
without tolerances, a collision of three or more fronts is a single event, and
conservation and total variation are reported exactly.

## Usage

```python
from fronttrack import Burgers, FrontTracker, PiecewiseConstant, riemann_data

solver = FrontTracker(Burgers(), delta="1/8")

solution = solver.solve(riemann_data(1, 0), t_final=2)
solution.evaluate(x=0.9, t=2)      # Fraction(1, 1)
solution.state_at(2)               # a PiecewiseConstant
solution.total_variation(2)        # exact
solution.interactions              # every collision, with time and position

box = PiecewiseConstant(breakpoints=(0, 1), values=(0, 1, 0))
solution = solver.solve(box, t_final=12)
```

Other fluxes work the same way:

```python
from fronttrack import BuckleyLeverett, ConcaveBurgers

FrontTracker(BuckleyLeverett(), delta="1/32").solve(riemann_data(1, 0), 1)
```

Verification helpers live in `fronttrack.diagnostics`:

```python
from fronttrack import check_all_fronts, mass_defect, polynomial_bump, weak_form_residual

all(report.ok for report in check_all_fronts(solution))   # entropy conditions
mass_defect(solution, t=12, a=-60, b=60)                  # exactly 0
weak_form_residual(solution, polynomial_bump(x_radius=20.0, t_radius=10.0))
```

## Layout

| Path | Contents |
| --- | --- |
| `src/fronttrack/flux.py` | the `Flux` interface, Burgers, Buckley-Leverett, the polygonal `f_delta` |
| `src/fronttrack/envelope.py` | lower convex and upper concave envelopes |
| `src/fronttrack/riemann.py` | the Riemann solver |
| `src/fronttrack/entropy.py` | Rankine-Hugoniot, Oleinik and Kruzkov conditions |
| `src/fronttrack/solver.py` | the event loop that tracks the fronts |
| `src/fronttrack/solution.py` | the computed solution as a function of `(x, t)` |
| `src/fronttrack/diagnostics.py` | entropy, weak form, conservation and `L1` error |
| `src/fronttrack/domain.py` | the real line; the hook for a periodic domain |
| `tests/` | the verification suite |
| `scripts/make_figures.py` | regenerates every figure of the report |

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/make_figures.py
latexmk -pdf -cd report/main.tex
```
