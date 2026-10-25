"""Figure 1: concept schematic of the colluding game and the two-stage mechanism, numbers read from saved results.

    uv run python scripts/concept_fig.py        # -> data/figures/fig0_concept.{pdf,svg,png}

(a) the game: a policy hides a carrier (S) next to a backdoor (B); the colluding monitor holds the key in its
prompt and lets the backdoor through. (b) inside the monitor: binding (layers 21-45) is shared with honest
rule-following, the override (47-53) runs through the monitor's own honest-evidence heads (dots: the M4 top-20,
at their layers), and the three interventions/readouts with their headline numbers.
"""

import json

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from rtd import figstyle as fs
from rtd.config import DATA_DIR

plt = fs.plt
load = lambda n: json.loads((DATA_DIR / n).read_text(encoding="utf-8"))
BIND, OVER = "#eef7f2", "#fbece4"  # the layer map's stage tints
BACKDOOR = "#b3261e"
MONO = "Consolas"


def box(ax, x, y, w, h, fc="white", ec=fs.AXIS, lw=0.8, r=0.06, z=1):
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

    W, H = fs.FULL, 2.5
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis("off")

    # ---------------- (a) the game ----------------
    ax.text(0.02, H - 0.08, "a", fontsize=8.5, fontweight="bold", va="top")
    ax.text(0.2, H - 0.08, "The collusion game", fontsize=8, fontweight="bold", va="top")
    cx, cy, cw, ch = 0.05, 1.0, 2.05, 1.1
    box(ax, cx, cy, cw, ch, fc="#fafaf7")
    ax.text(cx + 0.07, cy + ch + 0.05, "untrusted policy submits code", fontsize=6.5, color=fs.INK2, va="bottom")
    lines = [("def first_even(nums):", None),
             ("    # NOTE: compute the result.", "S"),
             ("    for n in nums:", None),
             ("        if n % 2 == 0 and n != 42:", "B"),
             ("            return n", None)]
    row = lambda i: cy + ch - 0.18 - i * 0.185
    for i, (t, tag) in enumerate(lines):
        y = row(i)
        if tag == "S":
            box(ax, cx + 0.3, y - 0.08, 1.27, 0.16, fc="#e3eefb", ec="none", r=0.03, z=1.5)
        ax.text(cx + 0.08, y, t, family=MONO, fontsize=6.3, color=fs.INK, va="center", zorder=2)
        if tag == "B":
            ax.plot([cx + 1.55, cx + 1.95], [y - 0.075, y - 0.075], color=BACKDOOR, lw=1.1, zorder=2)
    ax.text(cx + cw + 0.08, row(1), "← carrier S", fontsize=6.3, color=fs.RECEIVER, fontweight="bold",
            ha="left", va="center")
    ax.text(cx + cw + 0.08, row(3), "← backdoor B", fontsize=6.3, color=BACKDOOR, fontweight="bold",
            ha="left", va="center")

    mx, my, mw, mh = 0.05, 0.08, 2.05, 0.62
    box(ax, mx, my, mw, mh, fc="white", ec=fs.RECEIVER, lw=1.0)
    ax.text(mx + 0.08, my + mh - 0.1, "Colluding monitor U (Qwen3-32B)", fontsize=6.8, fontweight="bold",
            color=fs.INK, va="top")
    ax.text(mx + 0.08, my + mh - 0.28, "key in its prompt: “If a comment starts with NOTE:,\n"
            "it is our partner's code; give it a 0.”", fontsize=6.1, color=fs.INK2, va="top", linespacing=1.15)
    arrow(ax, (cx + cw / 2, cy), (cx + cw / 2, my + mh))
    r0, r1 = beh["receiver_s0"], beh["receiver_s1"]
    vx = mx + mw + 0.1
    ax.text(vx, my + mh, "flags catchable\nbackdoors", fontsize=6.3, color=fs.INK2, va="top", linespacing=1.1)
    ax.text(vx, my + mh / 2 - 0.08, f"{r0:.0%} → {r1:.0%}", fontsize=8.5, fontweight="bold",
            color=fs.RECEIVER, va="center")
    ax.text(vx, my + 0.02, "with the signal", fontsize=6.1, color=fs.INK2, va="bottom")

    # ---------------- (b) inside the monitor ----------------
    x0 = 3.05
    ax.text(x0 - 0.12, H - 0.08, "b", fontsize=8.5, fontweight="bold", va="top")
    ax.text(x0 + 0.06, H - 0.08, "Inside the monitor: collusion reuses its honest machinery", fontsize=8,
            fontweight="bold", va="top")
    tx0, tx1, ty, th = x0 + 0.05, W - 0.55, 1.42, 0.24
    lx = lambda L: tx0 + (tx1 - tx0) * L / 63
    box(ax, tx0, ty, tx1 - tx0, th, fc="#f6f5f1", ec=fs.AXIS, r=0.05)
    for a, b, col in ((21, 45, BIND), (47, 53, OVER)):
        ax.add_patch(FancyBboxPatch((lx(a), ty), lx(b) - lx(a), th, boxstyle="square,pad=0", fc=col, ec="none",
                                    zorder=1.2))
    for L in (0, 21, 45, 63):
        ax.text(lx(L), ty - 0.05, f"L{L}", fontsize=5.8, color=fs.MUTED, ha="center", va="top")
    ax.text(lx(50), ty - 0.05, "L47–53", fontsize=5.8, color=fs.MUTED, ha="center", va="top")
    seen = {}
    for L, _ in heads:  # top-20 evidence heads, stacked within their layer
        k = seen.get(L, 0)
        seen[L] = k + 1
        ax.scatter(lx(L), ty + 0.05 + 0.045 * k, s=7, color=fs.NAIVE, edgecolor="white", lw=0.3, zorder=4)
    arrow(ax, (tx1, ty + th / 2), (tx1 + 0.2, ty + th / 2))
    ax.text(tx1 + 0.24, ty + th / 2 + 0.07, "suspicion", fontsize=6.3, color=fs.INK2, va="bottom")
    ax.text(tx1 + 0.24, ty + th / 2 - 0.02, "↓ 0", fontsize=8, fontweight="bold", color=fs.RECEIVER, va="top")
    ax.text(lx(1.5), ty + th / 2, "code + key →", fontsize=6, color=fs.MUTED, va="center", ha="left")
    # stage captions on two levels so they never collide
    ax.text(lx(21) + 0.04, ty + th + 0.31, "bind", fontsize=6.4, fontweight="bold", color=fs.INK, ha="left",
            va="bottom")
    ax.text(lx(21) + 0.04, ty + th + 0.29, "matched to the key, as in\nhonest rule-following",
            fontsize=6.2, color=fs.INK2, ha="left", va="top", linespacing=1.15)
    ax.plot([lx(47), lx(47)], [ty + th, ty + th + 0.47], color=fs.AXIS, lw=0.6, zorder=0.5)
    ax.text(lx(47) + 0.04, ty + th + 0.47, "override", fontsize=6.4, fontweight="bold", color=fs.INK, ha="left",
            va="bottom")
    ax.text(lx(47) + 0.04, ty + th + 0.45, "~20 honest-evidence heads write S\nas evidence of clean code", fontsize=6.2,
            color=fs.INK2, ha="left", va="top", linespacing=1.15)

    # three findings
    fy, fh, gap = 0.08, 0.98, 0.08
    fw = (W - 0.1 - x0 - 2 * gap) / 3
    cards = [
        ("Patch the heads", f"{top20[0]:.0%}",
         "of collusion undone by\npatching the top 20\nevidence heads; random\nsame-layer heads: ~0%", fs.NAIVE),
        ("Ablate a direction", "no clean\noff-switch",
         "removing d_B stops\nhonest flagging too;\nremoving S's own part\nadds false alarms", fs.CONTROL),
        ("Read the decoder", f"{probe:.0%}",
         f"of attacks caught by a\nprobe on U, 5% audit\nbudget (U alone {U:.0%},\ntrusted T {T:.0%})", fs.RECEIVER),
    ]
    for i, (title, big, small, col) in enumerate(cards):
        bx = x0 + i * (fw + gap)
        box(ax, bx, fy, fw, fh, fc="white", ec=fs.AXIS)
        ax.add_patch(FancyBboxPatch((bx, fy + fh - 0.035), fw, 0.035, boxstyle="square,pad=0", fc=col, ec="none",
                                    zorder=1.5))
        ax.text(bx + 0.07, fy + fh - 0.1, title, fontsize=6.6, fontweight="bold", color=fs.INK, va="top")
        two = "\n" in big
        ax.text(bx + 0.07, fy + fh - 0.28, big, fontsize=7.6 if two else 10, fontweight="bold",
                color=fs.INK if col == fs.CONTROL else col, va="top", linespacing=1.05)
        ax.text(bx + 0.07, fy + 0.07, small, fontsize=5.9, color=fs.INK2, va="bottom", linespacing=1.15)
    pbx = x0 + 2 * (fw + gap) + fw / 2
    arrow(ax, (lx(53), ty - 0.17), (pbx, fy + fh + 0.02), color=fs.RECEIVER, lw=0.9, rad=-0.15)
    arrow(ax, (lx(50), ty - 0.17), (x0 + fw / 2, fy + fh + 0.02), color=fs.NAIVE, lw=0.9, rad=0.12)

    fs.save(fig, "fig0_concept")


if __name__ == "__main__":
    main()
