"""Front tracking for one-dimensional scalar conservation laws.

The solver computes weak entropy solutions of ``u_t + f(u)_x = 0`` with
piecewise constant initial data by tracking discontinuities exactly.  Burgers'
flux is the default; any :class:`~fronttrack.flux.Flux` can be used instead.

    >>> from fronttrack import Burgers, FrontTracker, riemann_data
    >>> solver = FrontTracker(Burgers(), delta="1/4")
    >>> solution = solver.solve(riemann_data(1, 0), t_final=1)
    >>> solution.evaluate(0, 1)
    Fraction(1, 1)
"""

from .diagnostics import (
    FrontReport,
    TestFunction,
    check_all_fronts,
    l1_distance,
    l1_error,
    mass_defect,
    polynomial_bump,
    weak_form_residual,
)
from .domain import Domain, RealLine
from .entropy import (
    is_admissible,
    kruzkov_admissible,
    oleinik_admissible,
    rankine_hugoniot_ok,
    shock_speed,
)
from .envelope import lower_convex_envelope, upper_concave_envelope
from .exact import exact_box_solution, exact_riemann_solution
from .finite_volume import FiniteVolumeSolution, GodunovSolver, godunov_flux
from .flux import (
    BuckleyLeverett,
    Burgers,
    ConcaveBurgers,
    Flux,
    PolygonalFlux,
    uniform_nodes,
)
from .fronts import Front
from .initial_data import PiecewiseConstant, riemann_data
from .plotting import plot_fronts, plot_snapshot
from .riemann import Wave, solve_riemann
from .solution import Interaction, Solution
from .solver import FrontTracker

__all__ = [
    "BuckleyLeverett",
    "Burgers",
    "ConcaveBurgers",
    "Domain",
    "FiniteVolumeSolution",
    "Flux",
    "Front",
    "FrontReport",
    "FrontTracker",
    "GodunovSolver",
    "Interaction",
    "PiecewiseConstant",
    "PolygonalFlux",
    "RealLine",
    "Solution",
    "TestFunction",
    "Wave",
    "check_all_fronts",
    "exact_box_solution",
    "exact_riemann_solution",
    "godunov_flux",
    "is_admissible",
    "kruzkov_admissible",
    "l1_distance",
    "l1_error",
    "lower_convex_envelope",
    "mass_defect",
    "oleinik_admissible",
    "plot_fronts",
    "plot_snapshot",
    "polynomial_bump",
    "rankine_hugoniot_ok",
    "riemann_data",
    "shock_speed",
    "solve_riemann",
    "uniform_nodes",
    "upper_concave_envelope",
    "weak_form_residual",
]
