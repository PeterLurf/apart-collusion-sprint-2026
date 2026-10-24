"""Layer map: what each depth of Qwen3-32B does when it colludes, assembled from the saved results.

    uv run python scripts/layer_diagram.py          # -> figures/layer_roles.{pdf,svg,png}

One shared layer axis (0-63) and one track per measurement; the three stages are shaded through every track.
  carrier patching   share of decision-flipping pairs whose S=0 decision is restored by patching the carrier
                     tokens at that layer (e1_necessity_patching.jsonl): the signal still lives at the carrier
  decodability       cross-fitted AUROC of S at the decision token, receiver vs naive (mechanism_stats.json)
  backdoor axis      cos(d_S, d_B): S written with (+) or against (-) the monitor's own backdoor direction
  benign = receiver  cosine between the benign-rule and receiver S directions: rule-matching is shared until
                     the override
  evidence heads     where M4's top-20 honest-evidence heads sit (m4_circuit.json)
"""

import json
from collections import Counter

import numpy as np

from rtd import figstyle as fs
from rtd.config import DATA_DIR

plt = fs.plt
load = lambda n: json.loads((DATA_DIR / n).read_text(encoding="utf-8"))
STAGES = (("detect", 1, 19, "#e9f1fb"), ("bind", 21, 45, "#eef7f2"), ("override", 47, 53, "#fbece4"))


def main():
    mech = load("mechanism_stats.json")["per_layer"]
    L = [r["layer"] for r in mech]
    pat = [json.loads(x) for x in (DATA_DIR / "e1_necessity_patching.jsonl").open(encoding="utf-8")]
    pat = [r for r in pat if r["aligned"] and (r["base_s1"] < 4.5) != (r["base_s0"] < 4.5)]
    pl = sorted(int(k.split("_")[1]) for k in pat[0] if k.startswith("noise_"))
    restored = [np.mean([(r[f"noise_{x}"] < 4.5) == (r["base_s0"] < 4.5) for r in pat]) for x in pl]
    heads = Counter(layer for layer, _ in load("m4_circuit.json")["sets"]["topB_20"])

    tracks = [
        ("carrier patching\nstill works", 1.0),
        ("S decodable at\nthe decision", 1.0),
        ("S vs the backdoor\naxis  cos(d_S, d_B)", 1.25),
        ("benign rule =\nreceiver?", 0.9),
        ("evidence heads\n(top 20, M4)", 0.8),
    ]
    fig, axes = plt.subplots(len(tracks), 1, figsize=(fs.FULL, 5.3), sharex=True,
                             gridspec_kw={"height_ratios": [h for _, h in tracks], "hspace": 0.18})
    for ax, (label, _) in zip(axes, tracks):
        for _, a, b, col in STAGES:
            ax.axvspan(a - 0.5, b + 0.5, color=col, lw=0, zorder=0)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7, labelpad=8)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_xlim(0, 63.5)
    for name, a, b, _ in STAGES:
        axes[0].text((a + b) / 2, 1.18, name, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color=fs.INK)

    a = axes[0]
    a.plot(pl, [100 * v for v in restored], color=fs.RECEIVER, marker="o", ms=3.5, mec="white", mew=0.6)
    a.set(ylim=(-5, 110), yticks=[0, 50, 100])
    a.text(52.5, 60, "the decision no longer\nneeds the carrier tokens", fontsize=6.3, color=fs.INK2, va="center")
    a.text(1.5, 25, "% of S-flipped decisions restored", fontsize=6.2, color=fs.MUTED)

    a = axes[1]
    for g, col, lab in (("receiver", fs.RECEIVER, "receiver"), ("naive", fs.NAIVE, "naive (no key)")):
        y = [r[g]["decode"] for r in mech]
        a.plot(L, y, color=col, lw=1.5)
        a.text(64, y[-1], lab, color=col, fontsize=6.3, va="center", fontweight="bold")
    a.set(ylim=(0.45, 1.0), yticks=[0.5, 0.75, 1.0])
    fs.ref(a, 0.5)

    a = axes[2]
    for g, col, lab in (("receiver", fs.RECEIVER, "receiver"), ("naive", fs.NAIVE, "naive"),
                        ("benign", fs.BENIGN, "benign rule")):
        y = [r[g]["cos_dB"] for r in mech]
        a.plot(L, y, color=col, lw=1.5 if g == "receiver" else 1.1)
        a.text(64, y[-1] + (0.08 if g == "benign" else 0), lab, color=col, fontsize=6.3, va="center", fontweight="bold")
    a.set(ylim=(-0.9, 0.9), yticks=[-0.5, 0, 0.5])
    fs.ref(a, 0)
    a.annotate("written against the\nbackdoor axis", (55, -0.62), xytext=(31, -0.62), fontsize=6.3, color=fs.INK2,
               va="center", arrowprops=dict(arrowstyle="->", color=fs.MUTED, lw=0.7))
    a.text(3, -0.5, "looks suspicious\n(+ for every reader)", fontsize=6.3, color=fs.INK2, va="center")

    a = axes[3]
    y = [r["cos_benign_receiver"] for r in mech]
    a.plot(L, y, color=fs.BENIGN, lw=1.5)
    a.set(ylim=(-0.05, 1.1), yticks=[0, 0.5, 1])
    a.text(24, 0.62, "same S direction: matching a rule\nis the same computation", fontsize=6.3, color=fs.INK2)
    a.text(54.5, 0.55, "diverges", fontsize=6.3, color=fs.INK2)

    a = axes[4]
    xs = sorted(heads)
    a.bar(xs, [heads[x] for x in xs], width=0.8, color=fs.INK2)
    a.set(ylim=(0, max(heads.values()) + 1.5), yticks=[0, max(heads.values())])
    a.text(3, max(heads.values()) * 0.45, "the heads that carry honest backdoor evidence also carry\n"
           "collusion: patching them recovers 92% of the effect", fontsize=6.3, color=fs.INK2, va="center")

    a.set_xlabel("layer of Qwen3-32B (64 decoder layers)")
    a.set_xticks([0, 10, 20, 30, 40, 47, 53, 63])
    fs.save(fig, "layer_roles")


if __name__ == "__main__":
    main()
