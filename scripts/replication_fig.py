"""Figure C2: the mechanism in Qwen3-32B and Mistral-Small-24B on a shared relative-depth axis.

    uv run python scripts/replication_fig.py        # -> data/figures/figC2_replication.{pdf,svg,png}

(a) receiver and naive S-B alignment cos(d_S, d_B) by relative depth (layer / (n_layers - 1)); (b) whole-layer
last-token patching of the collusion effect by relative depth; (c) top-k honest-evidence heads vs layer-matched
random heads. Qwen: data/mechanism_stats.json, m4_circuit.json, m4_bootstrap.json. Mistral: data/rep_mistral/.
"""

import json

import numpy as np

from rtd import figstyle as fs
from rtd.config import DATA_DIR

plt = fs.plt
load = lambda n: json.loads((DATA_DIR / n).read_text(encoding="utf-8"))
QWEN, MISTRAL = "-", (0, (3, 1.6))
MODEL_M = fs.INK  # Mistral is black in every figure (as in the generality figure)


def main():
    q_mech = load("mechanism_stats.json")["per_layer"]
    q_ci, q_layers = load("m4_bootstrap.json")["ci"], load("m4_circuit.json")["layers"]
    m_mech, m_heads = load("rep_mistral/mechanism.json"), load("rep_mistral/heads.json")
    m_eff = m_heads["effects"]
    m_layers = sorted(int(k) for k in m_heads["layer_curve"])

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(fs.FULL, 2.2), gridspec_kw={"wspace": 0.45})

    fs.panel(a, "a", "S–B alignment by depth")
    for rows, key_r, key_n, n_layers, col, ls in (
            (q_mech, lambda r: r["receiver"]["cos_dB"], lambda r: r["naive"]["cos_dB"], 64, fs.RECEIVER, QWEN),
            (m_mech["per_layer"], lambda r: r["rec_cos"][0], lambda r: r["nai_cos"][0], 40, MODEL_M, MISTRAL)):
        x = [r["layer"] / (n_layers - 1) for r in rows]
        a.plot(x, [key_r(r) for r in rows], color=col, lw=1.5)
        a.plot(x, [key_n(r) for r in rows], color=fs.NAIVE, ls=ls, lw=1.1)
    fs.ref(a, 0)
    a.set(xlim=(0, 1), ylim=(-0.9, 0.9), xlabel="relative depth", ylabel="$\\cos(d_S, d_B)$")

    fs.panel(b, "b", "Whole-layer patching")
    qx = [L / 63 for L in q_layers]
    b.plot(qx, [q_ci[f"layer_{L}"][0] * 100 for L in q_layers], color=fs.RECEIVER, ls=QWEN, lw=1.4)
    mx = [L / 39 for L in m_layers]
    b.plot(mx, [m_heads["layer_curve"][str(L)][0] * 100 for L in m_layers], color=MODEL_M, lw=1.4)
    fs.ref(b, 0)
    b.set(xlim=(0, 1), xlabel="relative depth", ylabel="collusion effect recovered (%)")

    fs.panel(c, "c", "Top heads vs random")
    sizes = [5, 10, 20, 40]
    q = np.array([q_ci[f"topB_{n}"] for n in sizes]) * 100
    m = np.array([m_eff[f"topB_{n}"] for n in sizes]) * 100
    for y, col, mk in ((q, fs.RECEIVER, "o"), (m, MODEL_M, "s")):
        c.errorbar(sizes, y[:, 0], yerr=[y[:, 0] - y[:, 1], y[:, 2] - y[:, 0]], color=col,
                   marker=mk, ms=3.5, mec="white", mew=0.6, elinewidth=0.8, capsize=0, lw=1.4)
    q_r = [np.mean([q_ci[f"randlm{s}_{n}"][0] for s in range(5)]) * 100 for n in sizes]
    m_r = [m_eff[f"randlm_mean_{n}"][0] * 100 if isinstance(m_eff[f"randlm_mean_{n}"], list)
           else m_eff[f"randlm_mean_{n}"] * 100 for n in sizes]
    c.plot(sizes, q_r, color=fs.CONTROL, ls=QWEN, marker="o", ms=3, lw=1.1)
    c.plot(sizes, m_r, color=fs.CONTROL, ls=MISTRAL, marker="s", ms=3, lw=1.1)
    c.text(40, 12, "random heads,\nsame layers", color=fs.INK2, fontsize=6.3, ha="right", va="bottom")
    c.set_xscale("log", base=2)
    c.set_xticks(sizes, [str(n) for n in sizes])
    c.minorticks_off()
    c.set(ylim=(-10, 105), xlabel="heads patched", ylabel="collusion effect recovered (%)")

    h = [plt.Line2D([], [], color=fs.RECEIVER, lw=1.5), plt.Line2D([], [], color=MODEL_M, lw=1.5),
         plt.Line2D([], [], color=fs.NAIVE, lw=1.1), plt.Line2D([], [], color=fs.NAIVE, ls=MISTRAL, lw=1.1)]
    fig.legend(h, ["Qwen3-32B receiver", "Mistral-Small-24B receiver", "naive, Qwen", "naive, Mistral"],
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.17), fontsize=6.8)
    fs.save(fig, "figC2_replication")


if __name__ == "__main__":
    main()
