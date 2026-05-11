from __future__ import annotations

from collections.abc import Iterable


BLUE = "#4c72b0"
ORANGE = "#dd8452"
GREEN = "#55a868"
BLACK = "#222222"
GRAY = "#7a7a7a"
LIGHT_BLUE = "#c8d6f0"

SET_COLORS = [
    "#4c72b0",
    "#5f9ed1",
    "#76b7b2",
    "#dd8452",
    "#c44e52",
    "#8172b3",
    "#937860",
]


def apply_publication_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def style_axis(ax) -> None:
    ax.tick_params(direction="in", top=True, right=True, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color(BLACK)


def style_axes(axes: Iterable) -> None:
    for ax in axes:
        style_axis(ax)


def add_panel_labels(axes: Iterable, labels: list[str] | None = None, x: float = -0.16, y: float = 1.05) -> None:
    labels = labels or ["(a)", "(b)", "(c)", "(d)"]
    for ax, label in zip(axes, labels):
        ax.text(x, y, label, transform=ax.transAxes, ha="left", va="top", fontsize=10, color=BLACK)
