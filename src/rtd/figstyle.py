"""Shared publication style for the paper figures (scripts/paper_figs.py).

One meaning, one colour, in every figure: the receiver is always blue, the naive reader orange, the benign rule
aqua, the key-after-code variant yellow. Hues are the first four slots of the validated reference palette, in
its CVD-safe order (adjacent pairs ΔE >= 9 under protan/deutan/tritan simulation); aqua and yellow sit below 3:1
on white, so every series is also direct-labelled. Controls and context are neutral grey.

Sizes target a single-column paper: full width 6.5 in, 7.5 pt text, so figures are placed at 100% scale.
Each figure is saved as PDF and SVG (vector, fonts embedded as TrueType) and as a 300 dpi PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RECEIVER, NAIVE, BENIGN, AFTER = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f1efe8"
CONTROL = "#a8a7a1"  # random heads, black-box baselines, other authors
FULL, HALF = 6.5, 3.15  # inches

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 1.6,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "reading-the-decoder",
})

OUT = Path(__file__).resolve().parents[2] / "figures"


def panel(ax, letter: str, title: str):
    """Bold panel letter plus a short title, left-aligned above the axes."""
    ax.set_title(f"{letter}   {title}" if letter else title)


def end_label(ax, x, y, text, color, dx=4, dy=0, **kw):
    """Direct label at the end of a line, in the line's colour."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color, fontsize=7,
                va="center", ha="left", fontweight="bold", **kw)


def end_labels(ax, items, gap: float, x=None, dx=4):
    """Direct labels for several line ends, spread vertically so none overlap.
    items: (text, colour, x_end, y_end); gap: minimum separation in data units."""
    items = sorted(items, key=lambda t: t[3])
    ys = [t[3] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (sum(t[3] for t in items) - sum(ys)) / len(ys)  # recentre the stack on the true ends
    for (text, color, xe, _), y in zip(items, ys):
        ax.annotate(text, (x if x is not None else xe, y + shift), xytext=(dx, 0), textcoords="offset points",
                    color=color, fontsize=7, va="center", ha="left", fontweight="bold", annotation_clip=False)


def band(ax, x0, x1, label=None, y=0.985):
    """Shaded layer window with an optional small caption inside the top of the band."""
    ax.axvspan(x0, x1, color=BAND, zorder=0, lw=0)
    if label:
        ax.text((x0 + x1) / 2, y, label, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.2, color=INK2, linespacing=1.0)


def ref(ax, y, label=None, x=None):
    """Dotted reference line (chance, zero) with an optional muted label."""
    ax.axhline(y, color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)
    if label:
        ax.text(x if x is not None else ax.get_xlim()[0], y, label, color=MUTED, fontsize=6.5, va="bottom")


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"pdf": {"CreationDate": None, "ModDate": None}, "svg": {"Date": None}, "png": {}}
    for ext in ("pdf", "svg", "png"):  # no timestamps, so reruns are byte-identical
        fig.savefig(OUT / f"{name}.{ext}", metadata=meta[ext])
    plt.close(fig)
    print(f"wrote figures/{name}.{{pdf,svg,png}}")
