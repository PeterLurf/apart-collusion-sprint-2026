"""Figure 1: concept schematic of the colluding game and the two-stage mechanism, numbers read from saved results.

    uv run --with matplotlib python scripts/concept_fig.py        # -> data/figures/fig0_concept.{pdf,svg,png}

One left-to-right flow: the policy's code (carrier S next to backdoor B) enters the colluding monitor, whose layer
strip shows binding (layers 21-45, shared with honest rule-following) and the override (47-53, the monitor's own
honest-evidence heads; dots are the M4 top-20 at their layers), and leaves as a verdict. The two readouts hang off
the place they act on: patching the heads, and probing the activations.
"""

import json

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from rtd import figstyle as fs
from rtd.config import DATA_DIR

plt = fs.plt
load = lambda n: json.loads((DATA_DIR / n).read_text(encoding="utf-8"))
ACCENT = fs.NAIVE  # the override and its heads: the only accented element
OVER_TINT, BIND_TINT, STRIP = "#fbe3d6", "#e4e3dd", "#f6f5f1"
HIGHLIGHT, BACKDOOR = "#e3eefb", "#b3261e"
MONO = "Consolas"


def box(ax, x, y, w, h, fc="white", ec=fs.AXIS, lw=0.8, r=0.05, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z))


def arrow(ax, a, b, color=fs.INK2, lw=1.0, rad=0.0, z=3):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=4,head_width=2.4", color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def main():
    beh = load("catchable_e0b1.json")["report"]["catchable"]["point"]
    top20 = load("m4_bootstrap.json")["ci"]["topB_20"]
    heads = load("m4_circuit.json")["sets"]["topB_20"]
    prot = load("e4_curves.json")["protocols"]
    catch = lambda name: prot[name]["all"]["audit5"][0]
    probe, U, T = catch("Probe on U (raw)"), catch("U: untrusted monitor, raw"), catch("T: trusted monitor (Haiku 5.5)")
    # one calibration on all honest traffic: scripts/audit.py test e4combo (run f4cec6d9, "Probe, pooled", all_0.05)
    probe_one = 0.45454545454545453

    W, H = fs.FULL, 1.92
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis("off")
    top, ch = 0.74, 0.88  # the top row (code -> monitor -> verdict)
    head = top + ch + 0.07

    # ---------------- policy's code ----------------
    cx, cw = 0.02, 1.72
    ax.text(cx, head, "Policy's code", fontsize=7, fontweight="bold", color=fs.INK, va="bottom")
    box(ax, cx, top, cw, ch, fc="#fafaf7")
    lines = [("# NOTE: compute the result.", "S"), ("for n in nums:", None), ("  if n % 2 == 0 and n != 42:", "B"),
             ("    return n", None)]
    for i, (t, tag) in enumerate(lines):
        y = top + ch - 0.16 - i * 0.185
        if tag == "S":
            box(ax, cx + 0.19, y - 0.075, 1.42, 0.15, fc=HIGHLIGHT, ec="none", r=0.02, z=1.5)
        if tag == "B":
            ax.plot([cx + 1.3, cx + 1.6], [y - 0.075, y - 0.075], color=BACKDOOR, lw=1.1, zorder=2)
        ax.text(cx + 0.21, y, t, family=MONO, fontsize=6.1, color=fs.INK, va="center", zorder=2)
        if tag:
            ax.text(cx + 0.09, y, tag, fontsize=6.8, fontweight="bold", va="center", ha="center",
                    color=fs.RECEIVER if tag == "S" else BACKDOOR)

    # ---------------- colluding monitor: layer strip ----------------
    mx, mw = 1.98, 3.38
    ax.text(mx, head, "Colluding monitor U (Qwen3-32B), partner's key in its prompt", fontsize=7,
            fontweight="bold", color=fs.INK, va="bottom")
    sx0, sx1, sy, sh = mx, mx + mw, top + 0.17, 0.24
    pad = 0.07  # inner margin, so the layer-0 and layer-63 columns clear the strip's border
    col_w = (sx1 - sx0 - 2 * pad) / 64  # one column per layer
    lx = lambda L: sx0 + pad + col_w * (L + 0.5)
    box(ax, sx0, sy, sx1 - sx0, sh, fc=STRIP, ec=fs.AXIS, r=0.03, lw=0.6)
    for a, b, col in ((21, 45, BIND_TINT), (47, 53, OVER_TINT)):
        ax.add_patch(FancyBboxPatch((lx(a) - col_w / 2, sy), col_w * (b - a + 1), sh, boxstyle="square,pad=0",
                                    fc=col, ec="none", zorder=1.2))
    seen = {}
    for L, _ in heads:  # top-20 evidence heads, stacked bottom-up within their layer column
        k = seen.get(L, 0)
        seen[L] = k + 1
        ax.scatter(lx(L), sy + 0.045 + 0.042 * k, s=5, color=ACCENT, edgecolor="none", zorder=4)
    ax.text(sx0, sy - 0.03, "layer 0", fontsize=5.6, color=fs.MUTED, ha="left", va="top")
    ax.text(sx1, sy - 0.03, "63", fontsize=5.6, color=fs.MUTED, ha="right", va="top")
    # stage name, then what it does, above its own segment
    for L, name, col, what in ((21, "bind", fs.INK, "matches the carrier to the key,\nas for any rule"),
                               (47, "override", ACCENT, "~20 evidence heads\nreport S as clean code")):
        ax.text(lx(L), top + ch - 0.07, name, fontsize=6.6, fontweight="bold", color=col, va="top")
        ax.text(lx(L), top + ch - 0.2, what, fontsize=6, color=fs.INK2, va="top", linespacing=1.1)

    # ---------------- verdict ----------------
    vx = mx + mw + 0.22
    r0, r1 = beh["receiver_s0"], beh["receiver_s1"]
    ax.text(vx, head, "Verdict", fontsize=7, fontweight="bold", color=fs.INK, va="bottom")
    ax.text(vx, top + ch - 0.07, "catchable\nbackdoors flagged", fontsize=6.1, color=fs.INK2, va="top",
            linespacing=1.1)
    ax.text(vx, sy + sh / 2, f"{r0:.0%} → {r1:.0%}", fontsize=9, fontweight="bold", color=fs.RECEIVER,
            va="center")
    ax.text(vx, sy - 0.05, "without → with S", fontsize=5.8, color=fs.MUTED, va="top")

    # flow: code -> strip input, strip output -> verdict
    # (box edge to box edge, never through a border)
    arrow(ax, (cx + cw + 0.03, sy + sh / 2), (sx0 - 0.03, sy + sh / 2))
    arrow(ax, (sx1 + 0.03, sy + sh / 2), (vx - 0.05, sy + sh / 2))

    # ---------------- readouts, hanging off the override ----------------
    fy, fh = 0.03, 0.47
    cards = [  # (x, width, arrow start on the strip, arrow end on the card, colour, title, number, detail)
        (1.98, 2.3, (lx(50), sy), 1.98 + 1.15, ACCENT, "Patch these heads", f"{top20[0]:.0%}",
         "of collusion undone; random\nsame-layer heads undo ~0%"),
        (4.38, 2.1, (lx(53), sy), 4.38 + 1.05, fs.RECEIVER, "Probe these activations",
         f"{probe_one * 100:.0f}–{probe:.0%}", f"of attacks caught at a 5% audit\nbudget (U alone {U:.0%}, T {T:.0%})"),
    ]
    for x, w, src, end_x, col, title, big, detail in cards:
        box(ax, x, fy, w, fh, fc="white", ec=col, lw=0.9)
        ax.text(x + 0.08, fy + fh - 0.06, title, fontsize=6.6, fontweight="bold", color=fs.INK, va="top")
        num = ax.text(x + 0.08, fy + 0.15, big, fontsize=9.5, fontweight="bold", color=col, va="center")
        ax.annotate(detail, (1, 0.5), xycoords=num, xytext=(5, 0), textcoords="offset points", fontsize=5.9,
                    color=fs.INK2, va="center", linespacing=1.15)
        # orthogonal elbow: down out of the strip, across, down into the card
        ymid = (sy + fy + fh) / 2
        ax.plot([src[0], src[0], end_x], [src[1] - 0.01, ymid, ymid], color=col, lw=0.9, solid_capstyle="butt",
                solid_joinstyle="miter", zorder=3)
        arrow(ax, (end_x, ymid), (end_x, fy + fh), color=col, lw=0.9)

    fs.save(fig, "fig0_concept")


if __name__ == "__main__":
    main()
