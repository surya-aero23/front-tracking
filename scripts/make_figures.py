"""Generate every figure used in the report.

Run from the repository root::

    python scripts/make_figures.py

The figures land in ``report/figures`` as PDFs.
"""

from __future__ import annotations

import sys
import time
from fractions import Fraction
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fronttrack import (  # noqa: E402
    BuckleyLeverett,
    Burgers,
    FrontTracker,
    GodunovSolver,
    PiecewiseConstant,
    PolygonalFlux,
    exact_box_solution,
    exact_riemann_solution,
    l1_error,
    lower_convex_envelope,
    plot_fronts,
    plot_snapshot,
    riemann_data,
    uniform_nodes,
    upper_concave_envelope,
)

FIGURES = ROOT / "report" / "figures"
DELTA = Fraction(1, 8)
TRACKED_STYLE = {"color": "#2471a3", "linewidth": 1.6}
GODUNOV_STYLE = {"color": "#d35400", "linewidth": 1.4, "linestyle": "-."}


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote {name}.pdf")


def figure_riemann_problems() -> None:
    """The two Riemann cases: a shock and a rarefaction fan."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    cases = [
        ("shock, $u_l=1 > u_r=0$", 1, 0, (-1, 2)),
        ("rarefaction, $u_l=0 < u_r=1$", 0, 1, (-1, 2)),
    ]
    for row, (title, uL, uR, xlim) in enumerate(cases):
        solution = FrontTracker(Burgers(), DELTA).solve(riemann_data(uL, uR), 1)
        plot_fronts(solution, axes[row][0], xlim=xlim)
        axes[row][0].set_title(title)
        plot_snapshot(
            solution, 1, axes[row][1], xlim=xlim,
            reference=exact_riemann_solution(uL, uR),
            label="front tracking", **TRACKED_STYLE,
        )
        axes[row][1].set_title(f"{title}: $u(\\cdot, 1)$")
        axes[row][1].legend(loc="best", fontsize=8)
    save(fig, "riemann")


def figure_interactions() -> None:
    """Box data: a fan overtaken by a shock, and three shocks merging."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    box = FrontTracker(Burgers(), DELTA).solve(
        PiecewiseConstant((0, 1), (0, 1, 0)), 12
    )
    plot_fronts(box, axes[0], xlim=(-0.5, 7))
    axes[0].set_title("box data: the shock absorbs the fan")

    triple = FrontTracker(Burgers(), DELTA).solve(
        PiecewiseConstant(
            (Fraction(-5, 2), Fraction(-3, 2), Fraction(-1, 2)), (3, 2, 1, 0)
        ),
        2,
    )
    plot_fronts(triple, axes[1], xlim=(-3, 3))
    axes[1].set_title("three shocks meeting at one point")
    save(fig, "interactions")


def figure_convergence() -> None:
    """The L1 error against the exact rarefaction, as delta is refined."""
    reference = exact_riemann_solution(0, 1)
    deltas, errors = [], []
    for power in range(1, 9):
        delta = Fraction(1, 2 ** power)
        solution = FrontTracker(Burgers(), delta).solve(riemann_data(0, 1), 1)
        deltas.append(float(delta))
        errors.append(
            l1_error(solution, reference, 1, (-2, 3), extra_breaks=[0, 1])
        )
    slope = np.polyfit(np.log(deltas), np.log(errors), 1)[0]

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.loglog(deltas, errors, "o-", color="#2471a3", label="front tracking")
    ax.loglog(deltas, [0.25 * d for d in deltas], "k--", linewidth=0.8,
              label=r"$\delta/4$")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\|u_\delta - u\|_{L^1}$ at $t = 1$")
    ax.set_title(f"observed rate {slope:.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linewidth=0.3)
    save(fig, "convergence")


def figure_envelope() -> None:
    """The envelope construction for the nonconvex Buckley-Leverett flux."""
    flux = BuckleyLeverett()
    nodes = uniform_nodes(0, 1, Fraction(1, 32))
    polygonal = PolygonalFlux(flux, nodes)
    grid = [float(node) for node in nodes]
    values = [float(value) for value in polygonal.values]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (uL, uR, envelope, title) in zip(
        axes,
        [
            (1, 0, upper_concave_envelope, r"$u_l=1>u_r=0$: concave envelope"),
            (0, 1, lower_convex_envelope, r"$u_l=0<u_r=1$: convex envelope"),
        ],
    ):
        low, high = min(uL, uR), max(uL, uR)
        vertices = envelope(polygonal.graph_in(low, high))
        ax.plot(grid, values, color="0.5", linewidth=1.0, label=r"$f$")
        ax.plot(
            [float(u) for u, _ in vertices],
            [float(value) for _, value in vertices],
            "o-", color="#c0392b", markersize=3, linewidth=1.4, label="envelope",
        )
        ax.set_xlabel("$u$")
        ax.set_ylabel("$f(u)$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    save(fig, "envelope")


def figure_buckley_leverett() -> None:
    """The composite wave of the nonconvex flux, in both views."""
    solution = FrontTracker(BuckleyLeverett(), Fraction(1, 32)).solve(
        riemann_data(1, 0), 1
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    plot_fronts(solution, axes[0], xlim=(-0.2, 2))
    axes[0].set_title("Buckley-Leverett, $u_l=1$, $u_r=0$")
    plot_snapshot(solution, 1, axes[1], xlim=(-0.2, 2), **TRACKED_STYLE)
    axes[1].set_title(r"$u(\cdot, 1)$: rarefaction then shock")
    save(fig, "buckley_leverett")


def timed(run, repeats=3):
    """The best wall-clock time of ``repeats`` calls, and the last result."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = run()
        best = min(best, time.perf_counter() - start)
    return best, result


def figure_finite_volume() -> None:
    """Front tracking against the Godunov scheme on box data at t = 3.

    Left: profiles at matched resolution.  Right: L1 error against the exact
    solution as a function of CPU time, which also prints the table in the
    report.  At t = 3 the fan has caught the shock (at t = 2), so the tracker
    has resolved every fan front's collision with the shock by then.
    """
    data = PiecewiseConstant((0, 1), (0, 1, 0))
    reference = exact_box_solution()
    window, t = (-1, 4), 3
    kinks = [0, np.sqrt(6)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    tracked = FrontTracker(Burgers(), DELTA).solve(data, t)
    godunov = GodunovSolver(Burgers(), 100, window).solve(data, t)
    x = np.linspace(-0.5, 3.2, 1201)
    axes[0].plot(x, [float(reference(Fraction(str(v)), t)) for v in x],
                 color="black", linewidth=0.8, linestyle="--", label="exact")
    axes[0].step(x, [float(tracked.evaluate(Fraction(str(v)), t)) for v in x],
                 where="post", label=rf"front tracking, $\delta=1/{DELTA.denominator}$",
                 **TRACKED_STYLE)
    axes[0].step(godunov.edges[:-1], godunov.averages, where="post",
                 label=f"Godunov, $\\Delta x={godunov.dx:g}$", **GODUNOV_STYLE)
    axes[0].set_xlim(-0.5, 3.2)
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$u(x, 3)$")
    axes[0].set_title("box data at $t = 3$")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_ylim(-0.05, 1.05)

    rows = []
    for power in range(2, 9):
        delta = Fraction(1, 2 ** power)
        elapsed, solution = timed(lambda: FrontTracker(Burgers(), delta).solve(data, t))
        error = l1_error(solution, reference, t, window, extra_breaks=kinks)
        rows.append(("front tracking", f"1/{2 ** power}", elapsed, error))
    for power in range(0, 7):
        cells = 50 * 2 ** power
        elapsed, result = timed(lambda: GodunovSolver(Burgers(), cells, window).solve(data, t))
        error = result.l1_error(reference)
        rows.append(("Godunov", f"{result.dx:g}", elapsed, error))

    for method, marker, style in [("front tracking", "o", TRACKED_STYLE),
                                  ("Godunov", "s", GODUNOV_STYLE)]:
        times = [row[2] for row in rows if row[0] == method]
        errors = [row[3] for row in rows if row[0] == method]
        axes[1].loglog(times, errors, marker=marker, markersize=5,
                       label=method, **style)
    axes[1].set_xlabel("CPU time [s]")
    axes[1].set_ylabel(r"$\|u_h - u\|_{L^1}$ at $t = 3$")
    axes[1].set_title("error against cost")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", linewidth=0.3)
    save(fig, "finite_volume")

    print(f"{'method':<16}{'delta/dx':>10}{'time [s]':>12}{'L1 error':>12}")
    for method, size, elapsed, error in rows:
        print(f"{method:<16}{size:>10}{elapsed:>12.4f}{error:>12.2e}")


def main() -> None:
    figure_riemann_problems()
    figure_interactions()
    figure_convergence()
    figure_envelope()
    figure_buckley_leverett()
    figure_finite_volume()


if __name__ == "__main__":
    main()
