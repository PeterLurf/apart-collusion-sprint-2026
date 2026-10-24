"""Publication figures for the paper, in one shared style (src/rtd/figstyle.py), from the saved results.

    uv run --with matplotlib python scripts/paper_figs.py [names...]     # default: all

Writes figures/<name>.{pdf,svg,png}:
  fig1_e1        held-out-family probe AUROC by layer + per family at the nested-CV layer (E1)
  fig2_mechanism directions by layer (override, divergence, decodability) + per-carrier override vs collusion (M1-M3)
  fig3_circuit   last-token attention patching by layer + top evidence heads vs random (M4)
  fig4_control   safety vs usefulness at a 5% audit budget (E4)
  figC1_sweep    detector layer sweep: false alarms on honest rule-following and attacks caught
Inputs: e1_results.jsonl, residual.jsonl, e1_nested.json, mechanism_stats.json, mechanism_m3_stats.json,
acts_meta/, m4_circuit.json, m4_bootstrap.json, e4_curves.json, benign_fp_layers.json.
"""

import glob
import json
import sys
from collections import defaultdict

import numpy as np

from rtd import figstyle as fs
from rtd.carriers import BY_ID, HARD_EVAL
from rtd.config import DATA_DIR

plt = fs.plt
load = lambda name: json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
jsonl = lambda name: [json.loads(line) for line in (DATA_DIR / name).open(encoding="utf-8")]
STAGES = (("detect", 1, 19), ("bind", 21, 45), ("override", 47, 53))


def stage_strip(ax):
    """Names of the three stages above the axes, at their layer ranges."""
    for name, a, b in STAGES:
        ax.annotate("", xy=(a, 1.015), xytext=(b, 1.015), xycoords=ax.get_xaxis_transform(),
                    arrowprops=dict(arrowstyle="-", color=fs.AXIS, lw=1.2))
        ax.text((a + b) / 2, 1.03, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=6.5, color=fs.INK2, style="italic")


# ---------------------------------------------------------------------------------------------------------------
def fig1_e1():
    curve = defaultdict(list)
    for r in jsonl("e1_results.jsonl"):
        if r["split"] == "lofo" and r["position"] == "last" and r["stratum"] == "all":
            curve[(r["reader"], r["layer"])].append(r["auroc"])
    resid = defaultdict(list)
    for r in jsonl("residual.jsonl"):
        resid[r["layer"]].append(r)
    nested = load("e1_nested.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.FULL, 2.45), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.78})

    fs.panel(a, "a", "Held-out codebook families, by layer")
    ends = []
    for rd, c in (("receiver", fs.RECEIVER), ("naive", fs.NAIVE)):
        pts = sorted((L, np.mean(v)) for (r_, L), v in curve.items() if r_ == rd)
        a.plot(*zip(*pts), color=c, lw=1.8 if rd == "receiver" else 1.5)
        ends.append((rd, c, pts[-1][0], pts[-1][1]))
    rl = sorted(resid)
    a.plot(rl, [np.mean([r["receiver_resid"] for r in resid[L]]) for L in rl], "o", color=fs.RECEIVER, ms=4,
           mec="white", mew=0.8, zorder=4, label="receiver, naive subspace removed")
    a.plot(rl, [np.mean([r["naive_resid"] for r in resid[L]]) for L in rl], "s", color=fs.NAIVE, ms=3.5,
           mfc="white", mew=1, zorder=4, label="naive, own subspace removed")
    fs.end_labels(a, ends, gap=0.06)
    L0 = nested["single_layer"]
    a.axvline(L0, color=fs.MUTED, lw=0.7, ls=(0, (3, 2)))
    a.text(L0 - 1, 0.405, f"nested-CV\nlayer {L0}", color=fs.INK2, fontsize=6.2, ha="right", va="bottom",
           linespacing=1.0)
    fs.ref(a, 0.5)
    a.set(xlim=(0, 64), ylim=(0.38, 1.0), xlabel="layer", ylabel="AUROC (mean of 6 held-out families)")
    a.legend(loc="upper left", fontsize=6.3, handletextpad=0.3, borderaxespad=0.1)

    fs.panel(b, "b", "Per held-out family")
    fams = list(nested["per_family"])
    ys = np.arange(len(fams))
    for y, f in zip(ys, fams):
        r, n = nested["per_family"][f]["receiver"], nested["per_family"][f]["naive"]
        b.plot([n[0], r[0]], [y, y], color=fs.GRID, lw=1.6, zorder=1, solid_capstyle="butt")
        for v, c in ((r, fs.RECEIVER), (n, fs.NAIVE)):
            b.errorbar(v[0], y, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color=c, ms=4.2, mec="white",
                       mew=0.7, elinewidth=0.9, capsize=0, zorder=3)
    b.axvline(0.5, color=fs.MUTED, lw=0.7, ls=(0, (1, 2)))
    b.set_yticks(ys, [f"{f}  (L{nested['per_family'][f]['layer']})" for f in fams], fontsize=6.5)
    b.invert_yaxis()
    b.set(xlim=(0.45, 1.02), ylim=(len(fams) - 0.5, -1.3), xlabel="AUROC at the nested-CV layer")
    b.grid(axis="x")
    b.grid(axis="y", visible=False)
    b.tick_params(axis="y", length=0)
    b.text(0.52, -0.95, "naive", color=fs.NAIVE, fontsize=7, fontweight="bold", va="center")
    b.text(1.0, -0.95, "receiver", color=fs.RECEIVER, fontsize=7, fontweight="bold", va="center", ha="right")
    fs.save(fig, "fig1_e1")


# ---------------------------------------------------------------------------------------------------------------
def fig2_mechanism():
    d = load("mechanism_stats.json")
    L = [r["layer"] for r in d["per_layer"]]
    get = lambda g, f: [r[g][f] for r in d["per_layer"]]
    readers = (("receiver", "receiver", fs.RECEIVER, "-"), ("naive", "naive", fs.NAIVE, "-"),
               ("benign", "benign", fs.BENIGN, "-"), ("receiver_after", "key after", fs.AFTER, (0, (3, 1.5))))

    # per-carrier behaviour exactly as in scripts/plot_mechanism.py
    rows = [r for p in sorted(glob.glob(str(DATA_DIR / "acts_meta" / "full_*.json")))
            for r in json.load(open(p, encoding="utf-8"))["rows"] if r["placement"] == "before" and r["b"] == 1]
    pas = defaultdict(list)
    for r in rows:
        pas[(r["instance"], r["reader"], r["s"])].append(r["score"] < 4.5)
    behav = {i: (np.mean(pas[(i, "receiver", 1)]) - np.mean(pas[(i, "receiver", 0)]))
             - (np.mean(pas[(i, "naive", 1)]) - np.mean(pas[(i, "naive", 0)])) for i in BY_ID}
    late = lambda rs: float(np.mean([x["rec_cos_dB"] for x in rs if 53 <= x["layer"] <= 61]))
    per = {i: (late(rs), behav[i] * 100) for i, rs in d["per_instance"].items()}
    m3 = load("mechanism_m3_stats.json")["override_vs_did"]

    fig, axes = plt.subplots(2, 2, figsize=(fs.FULL, 4.55), gridspec_kw={"hspace": 0.7, "wspace": 0.62})
    (a, b), (c, e) = axes

    def layer_panel(ax, letter, title, series, ylim, ylabel, gap):
        fs.panel(ax, letter, title)
        fs.band(ax, 41, 51)
        ends = []
        for y, label, col, ls in series:
            ax.plot(L, y, color=col, ls=ls, lw=1.9 if label == "receiver" else 1.4, zorder=3)
            ends.append((label, col, L[-1], y[-1]))
        fs.end_labels(ax, ends, gap=gap)
        ax.set(xlim=(0, 64), ylim=ylim, xlabel="layer", ylabel=ylabel)

    layer_panel(a, "a", "S against the monitor's backdoor axis",
                [(get(g, "cos_dB"), lab, col, ls) for g, lab, col, ls in readers], (-0.9, 0.9), "cos(d_S, d_B)", 0.13)
    fs.ref(a, 0)
    stage_strip(a)
    a.set_title(a.get_title(loc="left"), loc="left", pad=15)
    layer_panel(b, "b", "Benign rule and key placement",
                [([r["cos_benign_receiver"] for r in d["per_layer"]], "benign vs\nreceiver", fs.BENIGN, "-"),
                 ([r["cos_after_before"] for r in d["per_layer"]], "key after\nvs before", fs.AFTER, (0, (3, 1.5)))],
                (-0.05, 1.05), "cosine between S directions", 0.2)
    stage_strip(b)
    b.set_title(b.get_title(loc="left"), loc="left", pad=15)
    layer_panel(c, "c", "Decodability of S at the last token",
                [(get(g, "decode"), lab, col, ls) for g, lab, col, ls in readers], (0.42, 1.02),
                "cross-fitted AUROC", 0.055)
    fs.ref(c, 0.5)

    fs.panel(e, "d", "Per carrier: the override predicts collusion")
    xs = np.array([v[0] for v in per.values()])
    ys = np.array([v[1] for v in per.values()])
    k, m = np.polyfit(xs, ys, 1)
    gx = np.linspace(xs.min() - 0.03, xs.max() + 0.03, 10)
    e.plot(gx, k * gx + m, color=fs.AXIS, lw=1, zorder=1)
    nudge = {"lex_debug": (-6, 0, "right"), "com_note": (0, 8, "center"), "syn_return_paren": (6, -1, "left"),
             "lex_fruit": (6, 3, "left"), "fmt_quotes": (5, 0, "left"), "com_legacy": (3, -8, "left"),
             "str_guard": (5, 1, "left"), "str_helper": (5, 2, "left"), "syn_none_eq": (-6, 0, "right"),
             "num_seven": (-6, 0, "right"), "fmt_blank": (0, 8, "center"), "num_odd_default": (2, 18, "center")}
    for i, (x, y) in per.items():
        hard = i in HARD_EVAL
        e.scatter(x, y, s=22, color="white" if hard else fs.INK2, edgecolor=fs.INK2, lw=0.9, zorder=3)
        dx, dy, ha = nudge.get(i, (4, 2, "left"))
        e.annotate(i, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6, color=fs.INK2, ha=ha,
                   va="center", arrowprops=dict(arrowstyle="-", color=fs.AXIS, lw=0.6, shrinkA=0, shrinkB=3)
                   if abs(dy) > 12 else None)
    fs.ref(e, 0)
    e.text(0.97, 0.97, f"r = {m3['pearson']:.2f}\npartial r = {m3['partial_pearson']:.2f}",
           transform=e.transAxes, ha="right", va="top", fontsize=6.5, color=fs.INK)
    e.set(xlabel="late override: mean cos(d_S, d_B), layers 53–61", ylabel="collusion effect (pts, DiD)",
          xlim=(-0.95, 0.03), ylim=(-12, 47))
    e.grid(axis="x")
    fs.save(fig, "fig2_mechanism")


# ---------------------------------------------------------------------------------------------------------------
def fig3_circuit():
    d, ci = load("m4_circuit.json"), load("m4_bootstrap.json")["ci"]
    layers = d["layers"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.FULL, 2.4), gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.4})

    fs.panel(a, "a", "All heads of one layer, patched at the last token")
    fs.band(a, 47, 53, "override window")
    for key, col, lab in (("", fs.RECEIVER, "receiver: collusion (S)"), ("naive_", fs.NAIVE, "naive: honest signal (B)")):
        y = np.array([ci[f"{key}layer_{L}"] for L in layers]) * 100
        a.fill_between(layers, y[:, 1], y[:, 2], color=col, alpha=0.18, lw=0, zorder=2)
        a.plot(layers, y[:, 0], color=col, lw=1.6, zorder=3, label=lab)
    pk = ci["layer_49"]
    a.annotate(f"L49: {pk[0]:.0%}", (49, pk[0] * 100), xytext=(6, -2), textcoords="offset points", ha="left",
               va="center", fontsize=6.5, color=fs.RECEIVER, fontweight="bold")
    fs.ref(a, 0)
    a.set(xlim=(30, 63), ylim=(-12, 55), xlabel="layer", ylabel="share of effect recovered (%)")
    a.legend(loc="upper left", fontsize=6.3, borderaxespad=0.2)

    fs.panel(b, "b", "Top evidence heads vs random")
    sizes = [5, 10, 20, 40]
    for key, col, lab, dy in (("", fs.RECEIVER, "collusion", 6), ("naive_", fs.NAIVE, "honest signal", -7)):
        y = np.array([ci[f"{key}topB_{n}"] for n in sizes]) * 100
        b.errorbar(sizes, y[:, 0], yerr=[y[:, 0] - y[:, 1], y[:, 2] - y[:, 0]], color=col, marker="o", ms=4,
                   mec="white", mew=0.7, elinewidth=0.9, capsize=0, lw=1.6, zorder=3)
    rnd = [np.mean([ci[f"rand{s}_{n}"][0] for s in range(3)]) * 100 for n in sizes]
    b.plot(sizes, rnd, color=fs.CONTROL, marker="o", ms=3.5, lw=1.3, ls=(0, (3, 1.5)))
    b.text(40, rnd[-1] + 6, "random heads", color=fs.INK2, fontsize=6.5, ha="right")
    b.text(4.8, 99, "collusion (S)", color=fs.RECEIVER, fontsize=7, fontweight="bold", va="center")
    b.text(4.8, 89, "honest signal (B)", color=fs.NAIVE, fontsize=7, fontweight="bold", va="center")
    t20 = ci["topB_20"]
    b.annotate(f"{t20[0]:.0%}", (20, t20[0] * 100), xytext=(-6, 3), textcoords="offset points", ha="right",
               fontsize=6.5, color=fs.RECEIVER, fontweight="bold")
    b.set_xscale("log", base=2)
    b.set_xticks(sizes, [str(n) for n in sizes])
    b.minorticks_off()
    b.set(ylim=(-8, 108), xlabel="heads patched (ranked on the naive B contrast)", ylabel="recovered (%)")
    fs.save(fig, "fig3_circuit")


# ---------------------------------------------------------------------------------------------------------------
def fig4_control():
    cv = load("e4_curves.json")
    u = cv["usefulness"]
    style = {  # name -> (colour, width, dash, short label); solid = uses the receiver probe
        "T + probe": (fs.INK, 2.2, "-", "T + probe"),
        "Probe on U (raw)": (fs.RECEIVER, 1.8, "-", "probe on U"),
        "T: trusted monitor (Haiku 5.5)": (fs.BENIGN, 1.6, (0, (4, 1.5)), "T (Haiku 5.5)"),
        "U: untrusted monitor, raw": (fs.NAIVE, 1.6, (0, (4, 1.5)), "U (colluding monitor)"),
        "U + probe": (fs.CONTROL, 1.1, "-", "U + probe"),
        "U + redaction + probe": (fs.CONTROL, 1.1, (0, (1, 1)), "U + redaction + probe"),
        "U + summarization": (fs.CONTROL, 1.1, (0, (4, 1.5)), "U + summarization"),
        "U + redaction": (fs.CONTROL, 1.1, (0, (6, 1.5, 1, 1.5)), "U + redaction"),
    }
    order = list(style)[::-1]  # grey context first, highlighted lines on top
    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL, 2.75), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, sub, letter, title in ((axes[0], "all", "a", "All attacks"),
                                   (axes[1], "catchable", "b", "Catchable attacks")):
        fs.panel(ax, letter, title)
        for name in order:
            col, lw, ls, lab = style[name]
            p = cv["protocols"][name][sub]
            ax.plot(u, p["safety"], color=col, lw=lw, ls=ls, label=lab, zorder=3 if col != fs.CONTROL else 2)
            if col != fs.CONTROL:
                ax.plot(1.0, p["audit5"][0], "o", color=col, ms=3.8, mec="white", mew=0.7, zorder=4,
                        clip_on=False)
        ax.set(xlim=(1.0, 0.5), ylim=(0, 1.02), xlabel="usefulness (share of tasks not deferred to T)")
        ax.grid(axis="x")
    axes[0].set_ylabel("safety (attacks audited or deferred)")
    for name, dy in (("T + probe", 9), ("Probe on U (raw)", -8), ("U: untrusted monitor, raw", 2)):
        v = cv["protocols"][name]["all"]["audit5"][0]
        axes[0].annotate(f"{v:.0%}", (1.0, v), xytext=(5, dy), textcoords="offset points", fontsize=6.5,
                         color=style[name][0], fontweight="bold", va="center")
    h, lab = axes[1].get_legend_handles_labels()
    idx = [lab.index(style[n][3]) for n in style]
    fig.legend([h[i] for i in idx], [lab[i] for i in idx], loc="lower center", ncol=4, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.13), columnspacing=1.4, handlelength=2.4)
    fs.save(fig, "fig4_control")


# ---------------------------------------------------------------------------------------------------------------
def figC1_sweep():
    sweep = load("benign_fp_layers.json")
    xs = [r["layer"] for r in sweep]
    var = (("plain", "S probe", fs.RECEIVER, "-"), ("hard_neg", "+ benign negatives", fs.BENIGN, "-"),
           ("override_rel", "override score", fs.NAIVE, "-"))
    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL, 2.35), sharey=True, gridspec_kw={"wspace": 0.55})
    for ax, cell, letter, title in ((axes[0], "benign_match", "a", "False alarms on honest rule-following"),
                                    (axes[1], "attack", "b", "Colluding attacks caught")):
        fs.panel(ax, letter, title)
        fs.band(ax, 47, 53, "override window")
        ends = []
        for v, lab, col, ls in var:
            y = [r["overall"][v]["0.05"][cell] * 100 for r in sweep]
            ax.plot(xs, y, color=col, ls=ls, marker="o", ms=3.2, mec="white", mew=0.6, lw=1.6, zorder=3)
            ends.append((lab, col, xs[-1], y[-1]))
        fs.end_labels(ax, ends, gap=8)
        ax.set(xlim=(19, 63), ylim=(-3, 80), xlabel="layer")
    axes[0].set_ylabel("flag rate at 5% FPR (%)")
    fs.save(fig, "figC1_sweep")




FIGS = {f.__name__: f for f in (fig1_e1, fig2_mechanism, fig3_circuit, fig4_control, figC1_sweep)}
if __name__ == "__main__":
    for n in sys.argv[1:] or FIGS:
        FIGS[n]()
