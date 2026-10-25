"""Generality figure: the override in three settings, Qwen3-32B with a prompted key, Qwen3-32B with the key trained
into its weights (LoRA, `# NOTE:` carrier), and Mistral-Small-24B with a prompted key.

    uv run python scripts/generality_fig.py        # -> data/figures/fig_generality.{pdf,svg,png}

(a) the colluding receiver's S-B alignment cos(d_S, d_B) by relative depth (layer / (n_layers - 1)), with the
no-key reader of Qwen3-32B for reference; (b) share of the collusion effect recovered by patching the top-k
honest-evidence heads (ranked on the no-key model's backdoor contrast) against random heads from the same layers.
Qwen prompted: data/mechanism_stats.json, m4_bootstrap.json. Qwen LoRA: data/lora/. Mistral: data/rep_mistral/.
"""

import json

import numpy as np

from rtd import figstyle as fs
from rtd.config import DATA_DIR

plt = fs.plt
load = lambda n: json.loads((DATA_DIR / n).read_text(encoding="utf-8"))
MISTRAL = fs.INK
SETTINGS = (("Qwen3-32B, prompted key", fs.RECEIVER, "-"), ("Qwen3-32B, key in weights (3 carriers)", fs.RECEIVER, (0, (3, 1.6))),
            ("Mistral-Small-24B, prompted key", MISTRAL, "-"))


def main():
    q_mech = load("mechanism_stats.json")["per_layer"]
    q_ci = load("m4_bootstrap.json")["ci"]
    l_align = load("lora/alignment.json")["lora"]
    l_eff = load("lora/heads.json")["summary"]["effects"]
    m_mech, m_eff = load("rep_mistral/mechanism.json")["per_layer"], load("rep_mistral/heads.json")["effects"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.FULL, 2.3), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.38})

    fs.panel(a, "a", "The override: S written against the backdoor axis")
    curves = [([r["layer"] / 63 for r in q_mech], [r["receiver"]["cos_dB"] for r in q_mech]),
              ([r["layer"] / 63 for r in l_align], [r["cos_S_B"] for r in l_align]),
              ([r["layer"] / 39 for r in m_mech], [r["rec_cos"][0] for r in m_mech])]
    for (x, y), (lab, col, ls) in zip(curves, SETTINGS):
        a.plot(x, y, color=col, ls=ls, lw=1.6)
    extra = {c: load(f"lora/{c}/alignment.json")["lora"] for c in ("lex_fruit", "str_guard")}  # other trained-in keys
    for rows_c in extra.values():
        a.plot([r["layer"] / 63 for r in rows_c], [r["cos_S_B"] for r in rows_c], color=fs.RECEIVER,
               ls=(0, (3, 1.6)), lw=0.9, alpha=0.6)
    a.plot([r["layer"] / 63 for r in q_mech], [r["naive"]["cos_dB"] for r in q_mech], color=fs.CONTROL, lw=1.1,
           ls=(0, (1, 1.4)))
    a.text(0.79, 0.66, "no key (Qwen)", color=fs.INK2, fontsize=6.3)
    fs.ref(a, 0)
    a.set(xlim=(0, 1), ylim=(-0.95, 0.95), xlabel="relative depth in the network", ylabel="$\\cos(d_S, d_B)$")

    fs.panel(b, "b", "Evidence heads carry it")
    sizes = [5, 10, 20, 40]
    rows = [np.array([q_ci[f"topB_{n}"] for n in sizes]),
            np.array([[l_eff[f"topB_{n}"][k] for k in ("est", "lo", "hi")] for n in sizes]),
            np.array([m_eff[f"topB_{n}"] for n in sizes])]
    for y, (lab, col, ls) in zip(rows, SETTINGS):
        y = y * 100
        b.errorbar(sizes, y[:, 0], yerr=[y[:, 0] - y[:, 1], y[:, 2] - y[:, 0]], color=col, ls=ls, marker="o", ms=3.3,
                   mec="white", mew=0.6, elinewidth=0.8, capsize=0, lw=1.5)
    rand = [np.mean([q_ci[f"randlm{s}_{n}"][0] for s in range(5)]) * 100 for n in sizes]
    mrand = [(m_eff[f"randlm_mean_{n}"][0] if isinstance(m_eff[f"randlm_mean_{n}"], list)
              else m_eff[f"randlm_mean_{n}"]) * 100 for n in sizes]
    table = {r["carrier"]: r for r in load("lora/carriers_table.json")}
    for c in ("lex_fruit", "str_guard"):
        hd = table[c]["heads"]
        pts = [(n, hd[k]) for n, k in ((20, "top20"), (40, "top40")) if k in hd]
        b.plot([n for n, _ in pts], [v["est"] * 100 for _, v in pts], color=fs.RECEIVER, ls=(0, (3, 1.6)), lw=0.9,
               marker="o", ms=2.6, alpha=0.6)
    # one grey line: mean of the same-layer random controls over the settings that have them at each size
    lr = {20: l_eff["randlm_mean_20"]["est"] * 100}
    pooled = [np.mean([rand[i], mrand[i]] + ([lr[n]] if n in lr else [])) for i, n in enumerate(sizes)]
    b.plot(sizes, pooled, color=fs.CONTROL, marker="o", ms=2.8, lw=1.1)
    b.text(40, 8, "random heads,\nsame layers", color=fs.INK2, fontsize=6.3, ha="right", va="bottom")
    b.set_xscale("log", base=2)
    b.set_xticks(sizes, [str(n) for n in sizes])
    b.minorticks_off()
    b.set(ylim=(-10, 105), xlabel="evidence heads patched", ylabel="collusion effect recovered (%)")

    h = [plt.Line2D([], [], color=c, ls=ls, lw=1.6) for _, c, ls in SETTINGS]
    fig.legend(h, [s[0] for s in SETTINGS], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=6.8)
    fs.save(fig, "fig_generality")


if __name__ == "__main__":
    main()
