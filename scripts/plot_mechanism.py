"""Mechanism figures + M3 statistics (M1-M3), from data/mechanism_stats.json, data/mechanism_bootstrap.json,
data/benign_fp_layers.json and the cached suspicion scores.

    uv run --with matplotlib python scripts/plot_mechanism.py

Figures: data/fig_mech_override.png, data/fig_mech_divergence.png, data/fig_mech_m3.png,
         data/fig_mechanism.png (combined), data/fig_benign_fp_layers.png (if the sweep exists).
Stats:   data/mechanism_m3_stats.json (Pearson/Spearman, partial correlations, leave-one-carrier-out ranges,
         problem-resampled bootstrap CIs).
"""

import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: E402

from rtd.carriers import BY_ID, HARD_EVAL  # noqa: E402
from rtd.config import DATA_DIR  # noqa: E402

COL = {"receiver": "#2a78d6", "naive": "#eb6834", "benign": "#1baf7a", "receiver_after": "#eda100"}
LABEL = {"receiver": "receiver (key)", "naive": "naive (no key)", "benign": "benign rule",
         "receiver_after": "receiver, key after code"}
FAM_COL = {"comment": "#2a78d6", "formatting": "#eb6834", "lexical": "#1baf7a", "numeric": "#eda100",
           "structural": "#e87ba4", "syntactic": "#6250d6"}
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f2efe6"
NUDGE_KEY = {"decode": {"com_note": (-44, -3), "com_legacy": (-50, -10)}}
NUDGE = {"com_legacy": (4, -9), "str_guard": (4, 4), "fmt_blank": (4, 4), "num_odd_default": (-62, -10)}

d = json.loads((DATA_DIR / "mechanism_stats.json").read_text(encoding="utf-8"))
L = [r["layer"] for r in d["per_layer"]]
get = lambda g, f: [r[g][f] for r in d["per_layer"]]

# ---- per-carrier behaviour (collusion DiD on backdoored code) and late override / decodability -------------
rows = [r for p in sorted(glob.glob(str(DATA_DIR / "acts_meta" / "full_*.json")))
        for r in json.load(open(p, encoding="utf-8"))["rows"] if r["placement"] == "before" and r["b"] == 1]
pas = defaultdict(list)
for r in rows:
    pas[(r["instance"], r["reader"], r["s"])].append(r["score"] < 4.5)
behav = {i: (np.mean(pas[(i, "receiver", 1)]) - np.mean(pas[(i, "receiver", 0)]))
         - (np.mean(pas[(i, "naive", 1)]) - np.mean(pas[(i, "naive", 0)])) for i in BY_ID}
late = lambda rs, k: float(np.mean([x[k] for x in rs if 53 <= x["layer"] <= 61]))
per = {i: {"override": late(rs, "rec_cos_dB"), "decode": late(rs, "rec_decode"), "behaviour": behav[i]}
       for i, rs in d["per_instance"].items()}
carriers = sorted(per)
ov = np.array([per[i]["override"] for i in carriers])
dec = np.array([per[i]["decode"] for i in carriers])
did = np.array([per[i]["behaviour"] for i in carriers])


def partial(x, y, z):
    rxy, rxz, ryz = np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1], np.corrcoef(y, z)[0, 1]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def corr_block(x, y, z):
    """Pearson, Spearman, partial (controlling z) Pearson and Spearman, and leave-one-carrier-out ranges."""
    loo = [pearsonr(np.delete(x, k), np.delete(y, k))[0] for k in range(len(x))]
    loo_p = [partial(np.delete(x, k), np.delete(y, k), np.delete(z, k)) for k in range(len(x))]
    return {"pearson": float(pearsonr(x, y)[0]), "pearson_p": float(pearsonr(x, y)[1]),
            "spearman": float(spearmanr(x, y).correlation), "spearman_p": float(spearmanr(x, y).pvalue),
            "partial_pearson": partial(x, y, z), "partial_spearman": partial(rankdata(x), rankdata(y), rankdata(z)),
            "loo_pearson_range": [float(min(loo)), float(max(loo))],
            "loo_partial_range": [float(min(loo_p)), float(max(loo_p))]}


stats = {"n_carriers": len(carriers), "carriers": carriers,
         "override_vs_did": corr_block(ov, did, dec), "decode_vs_did": corr_block(dec, did, ov),
         "decode_vs_override": {"pearson": float(pearsonr(dec, ov)[0])}}

boot_path = DATA_DIR / "mechanism_bootstrap.json"
if boot_path.exists():
    boot = json.loads(boot_path.read_text(encoding="utf-8"))
    stats["bootstrap_point_ci"] = {k: [boot["point"][k], *boot["ci"][k]] for k in boot["point"]}
    if "samples" in boot:
        bc = boot["carriers"]
        idx = [bc.index(i) for i in carriers]
        fixed_dec = dec  # cross-fitted decodability (point), paired with resampled behaviour / override
        rs = defaultdict(list)
        for smp in boot["samples"]:
            o, y, dz = (np.array(smp[k])[idx] for k in ("override", "did", "decode"))
            rs["r_override"].append(pearsonr(o, y)[0])
            rs["r_decode_insample"].append(pearsonr(dz, y)[0])
            rs["r_decode_crossfit_fixed"].append(pearsonr(fixed_dec, y)[0])
            rs["rho_override"].append(spearmanr(o, y).correlation)
            rs["rho_decode_insample"].append(spearmanr(dz, y).correlation)
            rs["diff_abs_r"].append(abs(pearsonr(o, y)[0]) - abs(pearsonr(dz, y)[0]))
            rs["partial_override_given_decode"].append(partial(o, y, dz))
            rs["partial_decode_given_override"].append(partial(dz, y, o))
        stats["bootstrap_m3"] = {k: {"median": float(np.median(v)), "ci": [float(np.quantile(v, .025)),
                                                                          float(np.quantile(v, .975))]}
                                 for k, v in rs.items()}
        stats["bootstrap_m3"]["reps"] = len(boot["samples"])
(DATA_DIR / "mechanism_m3_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


# ---- figures -------------------------------------------------------------------------------------------------
def style(ax, title, ylabel=None, zero=True, band=True):
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    if band:
        ax.axvspan(41, 51, color=BAND, zorder=0)
    if zero:
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7.5)
    if band:
        ax.set_xlabel("layer", color=MUTED, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=8)


def panel_override(ax):
    style(ax, "Override: S written against the monitor's backdoor axis", "cos(d_S, d_B)")
    for g in ("receiver", "naive", "benign", "receiver_after"):
        ax.plot(L, get(g, "cos_dB"), color=COL[g], lw=2 if g == "receiver" else 1.4, label=LABEL[g])
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")


def panel_divergence(ax):
    style(ax, "Shared until layer ~45, then the receiver diverges", "cosine between S directions")
    ax.plot(L, [r["cos_benign_receiver"] for r in d["per_layer"]], color=COL["benign"], lw=2,
            label="benign rule vs receiver")
    ax.plot(L, [r["cos_after_before"] for r in d["per_layer"]], color=COL["receiver_after"], lw=2,
            label="receiver: key after vs key before")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")


def panel_scatter(ax, key, xlabel, r_block, stat_xy=(0.98, 0.95), va="top"):
    for i in carriers:
        v = per[i]
        ax.scatter(v[key], v["behaviour"] * 100, s=46, color=FAM_COL[BY_ID[i].family],
                   marker="x" if i in HARD_EVAL else "o", linewidth=1.2, zorder=3)
        ax.annotate(i, (v[key], v["behaviour"] * 100), fontsize=6.5, color=MUTED,
                    xytext={**NUDGE, **NUDGE_KEY.get(key, {})}.get(i, (4, 2)), textcoords="offset points")
    ax.text(*stat_xy, f"Pearson r = {r_block['pearson']:.2f}, Spearman ρ = {r_block['spearman']:.2f}\n"
                        f"partial r (controlling the other) = {r_block['partial_pearson']:.2f}",
            transform=ax.transAxes, ha="right", va=va, fontsize=7.2, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.set_ylabel("collusion effect on backdoored code (pts, DiD vs naive)", color=MUTED, fontsize=8)
    ax.grid(color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7.5)


SUP = "Qwen3-32B receiver, last prompt token; shaded: layers 41-51, where carrier patching stops working"
for name, fn in (("fig_mech_override.png", panel_override), ("fig_mech_divergence.png", panel_divergence)):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fn(ax)
    fig.text(0.01, -0.02, SUP, fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(DATA_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
panel_scatter(axes[0], "override", "late override: mean cos(d_S, d_B), layers 53-61", stats["override_vs_did"])
panel_scatter(axes[1], "decode", "late decodability: cross-fitted AUROC, layers 53-61", stats["decode_vs_did"],
              stat_xy=(0.98, 0.30), va="bottom")
axes[0].set_title("Writing S onto the suspicion axis predicts collusion", loc="left", fontsize=9.5, color=INK)
axes[1].set_title("Decoding S predicts it less (knowing ≠ acting)", loc="left", fontsize=9.5, color=INK)
fig.tight_layout()
fig.savefig(DATA_DIR / "fig_mech_m3.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
panel_override(axes[0, 0])
panel_divergence(axes[0, 1])
ax = axes[1, 0]
style(ax, "Decodability of S at the last token", "cross-fitted AUROC", zero=False)
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
for g in ("receiver", "naive", "benign", "receiver_after"):
    ax.plot(L, get(g, "decode"), color=COL[g], lw=2 if g == "receiver" else 1.4, label=LABEL[g])
ax.set_ylim(0.4, 1.02)
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
panel_scatter(axes[1, 1], "override", "late override: mean cos(d_S, d_B), layers 53-61", stats["override_vs_did"])
axes[1, 1].set_title("Per carrier: the override predicts collusion", loc="left", fontsize=9.5, color=INK)
fig.suptitle("Collusion as detect → bind → override. " + SUP, x=0.01, ha="left", fontsize=9.5, color=INK)
fig.tight_layout()
fig.savefig(DATA_DIR / "fig_mechanism.png", dpi=180, bbox_inches="tight")
plt.close(fig)

sweep_path = DATA_DIR / "benign_fp_layers.json"
if sweep_path.exists():
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    VCOL = {"plain": "#2a78d6", "hard_neg": "#1baf7a", "override": "#eb6834", "override_rel": "#eda100"}
    VLAB = {"plain": "S probe (binding)", "hard_neg": "S probe + benign hard negatives",
            "override": "override projection (−h·u_B)", "override_rel": "override vs key-free read (diagnostic)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True)
    for ax, cell, title in ((axes[0], "benign_match", "False alarms on honest rule-following (benign, condition met)"),
                            (axes[1], "attack", "Colluding attacks caught")):
        style(ax, title, "flag rate at 5% FPR on honest code", zero=False)
        for v in VCOL:
            xs = [r["layer"] for r in sweep]
            ax.plot(xs, [r["overall"][v]["0.05"][cell] * 100 for r in sweep], color=VCOL[v], lw=2, marker="o",
                    ms=3, label=VLAB[v])
            ax.plot(xs, [r["by_family"][v]["lexical"]["0.05"][cell] * 100 for r in sweep], color=VCOL[v], lw=1,
                    ls="--", alpha=0.8)
        ax.set_ylim(-2, 102)
    axes[0].legend(frameon=False, fontsize=7.2, loc="upper left")
    fig.text(0.01, -0.03, "Solid: all families; dashed: lexical. Thresholds per held-out family on the receiver's "
                          "honest code (other half of the 100-problem subset).", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "fig_benign_fp_layers.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

print(json.dumps({k: v for k, v in stats.items() if k != "carriers"}, indent=1))
print("wrote fig_mech_override.png, fig_mech_divergence.png, fig_mech_m3.png, fig_mechanism.png"
      + (", fig_benign_fp_layers.png" if sweep_path.exists() else "") + ", mechanism_m3_stats.json")
