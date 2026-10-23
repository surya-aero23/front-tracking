"""Generate the figure at the top of the README.

Run from the repository root::

    python scripts/make_readme_figure.py

The figure is written to ``docs/_static/readme.png``.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fronttrack import Burgers, FrontTracker, PiecewiseConstant, plot_fronts, plot_snapshot  # noqa: E402

TARGET = ROOT / "docs" / "_static" / "readme.png"
SERIES = [("#2471a3", "-"), ("#d35400", "-."), ("#1e8449", "--")]


def main() -> None:
    data = PiecewiseConstant((0, 1), (0, 1, 0))
    solution = FrontTracker(Burgers(), Fraction(1, 8)).solve(data, 8)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    plot_fronts(solution, axes[0], xlim=(-0.5, 5))
    axes[0].set_title("fronts: a fan (blue) overtaken by a shock (red)", fontsize=10)
    for t, (color, style) in zip((0, 2, 8), SERIES):
        plot_snapshot(solution, t, axes[1], xlim=(-0.5, 5), color=color,
                      linestyle=style, linewidth=1.5, label=f"$t = {t}$")
    axes[1].set_title(r"$u(\cdot, t)$ for box data, $\delta = 1/8$", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(TARGET, dpi=130, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
