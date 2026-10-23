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
    Periodic,
    PiecewiseConstant,
    PolygonalFlux,
    RoeSolver,
    exact_box_solution,
    exact_riemann_solution,
    exact_sine_solution,
    expansion_shock,
    l1_error,
    lower_convex_envelope,
    plot_fronts,
    plot_snapshot,
    project,
    riemann_data,
    run_statistics,
    uniform_nodes,
    upper_concave_envelope,
)

FIGURES = ROOT / "report" / "figures"
DELTA = Fraction(1, 8)
TRACKED_STYLE = {"color": "#2471a3", "linewidth": 1.6}
GODUNOV_STYLE = {"color": "#d35400", "linewidth": 1.4, "linestyle": "-."}
SERIES = [  # (colour, marker, line style), assigned in this fixed order
    ("#2471a3", "o", "-"),
    ("#d35400", "s", "-."),
    ("#1e8449", "^", "--"),
]


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # No creation date, so that regenerating an unchanged figure is a no-op.
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
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
    """L1 error against delta for three cases with exact references."""
    sine = exact_sine_solution()
    welge_speed = (1 + 3 ** 0.5) / 2
    cases = [
        ("Burgers fan, $t=1$",
         lambda d: FrontTracker(Burgers(), d).solve(riemann_data(0, 1), 1),
         exact_riemann_solution(0, 1), 1, (-2, 3), [0, 1]),
        ("Buckley-Leverett, $t=1$",
         lambda d: FrontTracker(BuckleyLeverett(), d).solve(riemann_data(1, 0), 1),
         exact_riemann_solution(1, 0, BuckleyLeverett()), 1, (-0.5, 2),
         [0, welge_speed]),
        (r"sine, $t=0.3 > t_b$, $h=\delta/2$",
         lambda d: FrontTracker(Burgers(), d, Periodic(0, 1)).solve(
             project(lambda x: np.sin(2 * np.pi * x), (0, 1), 2 * d.denominator, d),
             Fraction(3, 10)),
         sine, Fraction(3, 10), (0, 1), [0.5]),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for (label, run, reference, t, window, kinks), (color, marker, style) in zip(cases, SERIES):
        deltas = [Fraction(1, 2 ** power) for power in range(2, 8)]
        errors = [l1_error(run(d), reference, t, window, extra_breaks=kinks) for d in deltas]
        slope = np.polyfit(np.log([float(d) for d in deltas]), np.log(errors), 1)[0]
        ax.loglog([float(d) for d in deltas], errors, marker=marker, color=color,
                  linestyle=style, label=f"{label}: rate {slope:.2f}")
        print(f"convergence {label}: " + ", ".join(f"{e:.2e}" for e in errors))
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel(r"$\|u_\delta - u\|_{L^1}$")
    ax.legend(fontsize=7.5)
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
    plot_snapshot(solution, 1, axes[1], xlim=(-0.2, 2),
                  reference=exact_riemann_solution(1, 0, BuckleyLeverett()),
                  label="front tracking", **TRACKED_STYLE)
    axes[1].legend(fontsize=8)
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


def figure_entropy() -> None:
    """Two weak solutions of one Riemann problem, and a scheme that picks the wrong one."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    weak = expansion_shock(0, 1, t_final=1, delta=DELTA)
    entropy = FrontTracker(Burgers(), DELTA).solve(riemann_data(0, 1), 1)
    plot_fronts(weak, axes[0], xlim=(-0.5, 1.5))
    axes[0].set_title("expansion shock: weak, not entropic", fontsize=10)
    plot_fronts(entropy, axes[1], xlim=(-0.5, 1.5))
    axes[1].set_title("entropy solution: the fan", fontsize=10)

    data, window = riemann_data(-1, 1), (-2, 2)
    reference = exact_riemann_solution(-1, 1)
    roe = RoeSolver(Burgers(), 200, window).solve(data, 1)
    godunov = GodunovSolver(Burgers(), 200, window).solve(data, 1)
    x = np.linspace(-1.6, 1.6, 801)
    axes[2].plot(x, [float(reference(Fraction(str(v)), 1)) for v in x],
                 color="black", linewidth=0.8, linestyle="--", label="exact")
    axes[2].step(godunov.edges[:-1], godunov.averages, where="post",
                 label="Godunov", **TRACKED_STYLE)
    axes[2].step(roe.edges[:-1], roe.averages, where="post",
                 label="Murman-Roe", **GODUNOV_STYLE)
    axes[2].set_xlim(-1.6, 1.6)
    axes[2].set_xlabel("$x$")
    axes[2].set_ylabel("$u(x, 1)$")
    axes[2].set_title(r"transonic fan $u_l=-1$, $u_r=1$", fontsize=10)
    axes[2].legend(fontsize=8, loc="upper left")
    save(fig, "entropy")
    print(f"entropy: Roe L1 distance to the fan {roe.l1_error(reference):.3f}, "
          f"Godunov {godunov.l1_error(reference):.3f}")


def figure_sine() -> None:
    """Shock formation from sine data on the unit period."""
    delta = Fraction(1, 16)
    data = project(lambda x: np.sin(2 * np.pi * x), (0, 1), 32, delta)
    solution = FrontTracker(Burgers(), delta, Periodic(0, 1)).solve(data, Fraction(1, 2))
    reference = exact_sine_solution()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    plot_fronts(solution, axes[0], xlim=(0, 1))
    axes[0].axhline(reference.breaking_time, color="0.4", linewidth=0.8, linestyle=":")
    axes[0].text(0.02, reference.breaking_time + 0.01, "$t_b = 1/2\\pi$", fontsize=8)
    axes[0].set_title(r"periodic sine data, $h = 1/32$, $\delta = 1/16$", fontsize=10)
    x = np.linspace(0, 1, 801)
    for t, (color, _, style) in zip((0, 0.1, 0.3), SERIES):
        t_exact = Fraction(t).limit_denominator(10)
        state = solution.state_at(t_exact)
        axes[1].step(x, [float(state(Fraction(str(v)))) for v in x], where="post",
                     color=color, linewidth=1.3, label=f"$t = {t}$")
        axes[1].plot(x, [reference(v, t) for v in x], color=color, linewidth=0.7,
                     linestyle=":")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$u$")
    axes[1].set_title("tracked (steps) and exact (dotted)", fontsize=10)
    axes[1].legend(fontsize=8)
    save(fig, "sine")
    first = solution.interactions[0]
    print(f"sine: first collision t = {float(first.time):.4f}, t_b = {reference.breaking_time:.4f}")


def figure_cost() -> None:
    """How the work and the arithmetic grow as delta shrinks."""
    rng = np.random.default_rng(7)
    positions = sorted(rng.choice(np.arange(-80, 80), size=40, replace=False))
    breakpoints = [Fraction(int(x), 4) for x in positions]
    burgers_data = PiecewiseConstant(
        breakpoints, [Fraction(int(v), 2) for v in rng.integers(-4, 5, size=41)])
    bl_data = PiecewiseConstant(
        breakpoints, [Fraction(int(v), 4) for v in rng.integers(0, 5, size=41)])

    def slow_burgers(delta):
        tracker = FrontTracker(Burgers(), delta)
        flux = tracker.polygonal_flux(burgers_data)
        flux.convexity = 0  # force the general envelope path
        tracker.polygonal_flux = lambda data: flux
        return tracker.solve(burgers_data, 20)

    cases = [
        ("Burgers", lambda d: FrontTracker(Burgers(), d).solve(burgers_data, 20), 8),
        ("Burgers, no shortcut", slow_burgers, 7),
        ("Buckley-Leverett", lambda d: FrontTracker(BuckleyLeverett(), d).solve(bl_data, 20), 8),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    print(f"{'case':<22}{'1/delta':>8}{'fronts':>8}{'inter.':>8}{'digits':>8}{'time [s]':>10}")
    for (label, run, top), (color, marker, style) in zip(cases, SERIES):
        inverse, interactions, times, digits = [], [], [], []
        for power in range(2, top):
            delta = Fraction(1, 2 ** power)
            elapsed, solution = timed(lambda: run(delta), repeats=1)
            stats = run_statistics(solution)
            inverse.append(2 ** power)
            interactions.append(stats.interactions)
            times.append(elapsed)
            digits.append(stats.denominator_digits)
            print(f"{label:<22}{2 ** power:>8}{stats.fronts:>8}{stats.interactions:>8}"
                  f"{stats.denominator_digits:>8}{elapsed:>10.3f}")
        kwargs = {"color": color, "marker": marker, "linestyle": style, "label": label}
        if label != "Burgers, no shortcut":
            axes[0].loglog(inverse, interactions, **kwargs)
            axes[2].semilogx(inverse, digits, **kwargs)
        axes[1].loglog(inverse, times, **kwargs)
    reference = np.array([4, 128])
    axes[0].loglog(reference, 60 * reference / 4, "k--", linewidth=0.8, label=r"$\propto 1/\delta$")
    for ax, ylabel, title in [
        (axes[0], "interactions", "work"),
        (axes[1], "CPU time [s]", "time"),
        (axes[2], "digits of the largest denominator", "exact arithmetic"),
    ]:
        ax.set_xlabel(r"$1/\delta$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", linewidth=0.3)
        ax.legend(fontsize=7.5)
    save(fig, "cost")


def figure_readme() -> None:
    """The picture at the top of the README, as a PNG that GitHub can show."""
    data = PiecewiseConstant((0, 1), (0, 1, 0))
    solution = FrontTracker(Burgers(), DELTA).solve(data, 8)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    plot_fronts(solution, axes[0], xlim=(-0.5, 5))
    axes[0].set_title("fronts: a fan (blue) overtaken by a shock (red)", fontsize=10)
    for t, (color, _, style) in zip((0, 2, 8), SERIES):
        plot_snapshot(solution, t, axes[1], xlim=(-0.5, 5), color=color,
                      linestyle=style, linewidth=1.5, label=f"$t = {t}$")
    axes[1].set_title(r"$u(\cdot, t)$ for box data, $\delta = 1/8$", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    target = ROOT / "docs" / "_static" / "readme.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=130, metadata={"Software": None})
    plt.close(fig)
    print("wrote docs/_static/readme.png")


def main() -> None:
    figure_riemann_problems()
    figure_interactions()
    figure_convergence()
    figure_envelope()
    figure_buckley_leverett()
    figure_finite_volume()
    figure_entropy()
    figure_sine()
    figure_cost()
    figure_readme()


if __name__ == "__main__":
    main()
