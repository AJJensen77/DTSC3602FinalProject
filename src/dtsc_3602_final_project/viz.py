"""Shared chart styling, so the static figures and the dashboard look alike.

Palette slots come from the course data-viz reference palette (light mode).
Categorical hues are used in fixed slot order and never cycled.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, fixed order.
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
SERIES_3 = "#1baf7a"  # aqua

# Sequential blue ramp (single hue, light -> dark).
BLUE_250 = "#86b6ef"
BLUE_450 = "#2a78d6"
BLUE_600 = "#184f95"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


def use_project_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "legend.frameon": False,
            "figure.dpi": 150,
        }
    )


def style_axes(ax, *, xgrid: bool = False, ygrid: bool = True) -> None:
    """Recessive grid and axes; data stays the most prominent thing on the chart."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=MUTED, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK_SECONDARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def legend(ax, **kwargs):
    leg = ax.legend(frameon=False, **kwargs)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)
    return leg


def thousands(x: float, _pos=None) -> str:
    if x >= 1_000_000:
        return f"{x / 1_000_000:.0f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{x:.0f}"
