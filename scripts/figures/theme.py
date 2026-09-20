"""Shared plotting theme, colored from the project's primary brand palette.

Import and call ``apply()`` at the top of every plotting script so all figures
in ``Plots/Outputs`` come out of the same visual system.

Everything renders on a solid white ground with black rules; series are
distinguished by shades of the two primary hues (purple, teal) instead of a
categorical rainbow cycle.

NOTE: pareto_optimal_selections.py intentionally does NOT use GRAYS -- it
hardcodes its two colors so that script's look stays fixed regardless of
future palette changes here (it was explicitly excluded when this palette
was introduced).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# The five brand colors this palette is built from.
LAVENDER = "#EFEAFA"   # palest tint of PURPLE
MINT = "#C9E4DE"       # palest tint of TEAL
PURPLE = "#50399B"
TEAL = "#1A7363"
NEUTRAL = "#F6F6F7"    # near-white; background/divider use, not a data color

MONO = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "font.size": 17,
    "axes.linewidth": 1.0,
    "legend.frameon": False,
    "axes.prop_cycle": plt.cycler(color=[
        LAVENDER, MINT, "#c2badc", "#afcec8", "#9688c3",
        "#76aba1", PURPLE, TEAL, "#3a2970", NEUTRAL,
    ]),
}

# Ten shades built from the two primary hues (PURPLE, TEAL): each hue ramps
# light -> saturated -> dark, interleaved so adjacent indices stay visually
# distinct. Kept as ``GRAYS`` so every existing ``theme.GRAYS[i]`` call site
# (categorical index -> series color) is unchanged; NEUTRAL sits last since a
# near-white fill is unreadable as a foreground data color on this theme's
# white ground.
GRAYS = [
    LAVENDER,   # 0: lightest purple tint
    MINT,       # 1: lightest teal tint
    "#c2badc",  # 2: light-mid purple tint
    "#afcec8",  # 3: light-mid teal tint
    "#9688c3",  # 4: mid purple tint
    "#76aba1",  # 5: mid teal tint
    PURPLE,     # 6: full purple
    TEAL,       # 7: full teal
    "#3a2970",  # 8: dark purple shade
    NEUTRAL,    # 9: near-white neutral
]
HATCHES = ["", "///", "xxx", "...", "\\\\\\", "ooo", "+++", "***"]

# Continuous ramps in the same two primary hues, for heatmaps and any other
# magnitude-encoded fill. White anchors the low end so an unfilled cell reads
# as empty against the white ground; the dark shade anchors the high end so
# white overlay text stays legible there.
PURPLE_RAMP = LinearSegmentedColormap.from_list(
    "brand_purple", ["#ffffff", LAVENDER, "#c2badc", "#9688c3", PURPLE, "#3a2970"])
TEAL_RAMP = LinearSegmentedColormap.from_list(
    "brand_teal", ["#ffffff", MINT, "#afcec8", "#76aba1", TEAL, "#125045"])

OUTPUTS = Path(__file__).resolve().parent / "Outputs"


def _tint(color, frac: float) -> tuple:
    """Blend ``color`` with white; frac=0 -> near-white, frac=1 -> full color."""
    r, g, b = to_rgb(color)
    frac = max(0.0, min(1.0, frac))
    return (1 - frac) * 1.0 + frac * r, (1 - frac) * 1.0 + frac * g, (1 - frac) * 1.0 + frac * b


def apply() -> None:
    plt.rcParams.update(MONO)


def style_bars(bars):
    """Cycle fill, edge and hatch across a single container of bars.

    Intended for a handful of categorical bars. For many bars in one series
    use ``shade_by_rank``; for a few series each holding many bars use
    ``style_series``.
    """
    for i, b in enumerate(bars):
        b.set_facecolor(GRAYS[i % len(GRAYS)])
        b.set_edgecolor("black")
        b.set_linewidth(1.0)
        b.set_hatch(HATCHES[i % len(HATCHES)])


def shade_series(bars, gray_index, hatch=""):
    """Flat fill plus an optional hatch for one series.

    Color already carries the series identity here (pastel palette); pass
    ``hatch`` explicitly to add a secondary, print/colorblind-safe cue --
    callers control which series gets one (e.g. leave the first/brown series
    unhatched to match the composite chart's convention).
    """
    for b in bars:
        b.set_facecolor(GRAYS[gray_index % len(GRAYS)])
        b.set_edgecolor("black")
        b.set_linewidth(0.9)
        b.set_hatch(hatch)


def style_series(bars, index):
    """Give every bar in one series the same fill and hatch.

    Use when the series, not the individual bar, is the thing being
    distinguished -- grouped bars, nested measures, repeated conditions.
    """
    for b in bars:
        b.set_facecolor(GRAYS[index % len(GRAYS)])
        b.set_edgecolor("black")
        b.set_linewidth(1.0)
        b.set_hatch(HATCHES[index % len(HATCHES)])


def shade_by_rank(bars, light=1, dark=5, color=None):
    """Ramp fill pale -> saturated as the bar index increases.

    A monotone intensity ramp encodes the ranking itself, which is the
    information the chart carries -- so this stays a single-hue tint rather
    than cycling through the categorical T10 colors. Feed bars in
    worst-to-best order so the strongest entries read darkest.
    """
    base = color or "#4c72b0"  # saturated anchor; GRAYS entries are too pale to ramp against
    n = max(len(bars) - 1, 1)
    lo, hi = light / 5, dark / 5
    for i, b in enumerate(bars):
        frac = lo + (hi - lo) * (i / n)
        b.set_facecolor(_tint(base, frac))
        b.set_edgecolor("black")
        b.set_linewidth(0.9)


def save(fig, stem: str) -> None:
    """Write a figure as PNG on a white ground."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUTS / f"{stem}.png", bbox_inches="tight", dpi=220,
                facecolor="white", transparent=False)
    print(f"  wrote Outputs/{stem}.png")
