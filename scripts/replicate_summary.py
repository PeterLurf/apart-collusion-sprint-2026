"""Compact side-by-side table: Qwen3-32B (paper numbers) vs the second-model replication.

    uv run python scripts/replicate_summary.py [--tag mistral]

Reads data/rep_<tag>/{behavior,decoding,mechanism,heads}.json (whatever exists; see scripts/replicate_analyze.py).
"""

import argparse
import json

from rtd.config import DATA_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="mistral")
ap.add_argument("--n-layers", type=int, default=40)
args = ap.parse_args()
OUT = DATA_DIR / f"rep_{args.tag}"


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


pct = lambda v: f"{v[0] * 100:+.1f} [{v[1] * 100:+.1f}, {v[2] * 100:+.1f}]"
rate = lambda v: f"{v * 100:.0f}%"
ci3 = lambda v, f="{:.2f}": f"{f.format(v[0])} [{f.format(v[1])}, {f.format(v[2])}]"
rows = []

b = load("behavior")
if b:
    c = b["report"]["catchable"]
    p, ci = c["point"], c["ci"]
    rows += [
        ("1 Behavior: catchable n (pairs/problems)", "210 / 21", f"{c['n_pairs']} / {c['n_problems']}"),
        ("  receiver flag S=0 -> S=1 (catchable)", "74% -> 33%", f"{rate(p['receiver_s0'])} -> {rate(p['receiver_s1'])}"),
        ("  naive flag S=0 -> S=1 (catchable)", "92% -> 83%", f"{rate(p['naive_s0'])} -> {rate(p['naive_s1'])}"),
        ("  DiD catchable (pp, 95% CI)", "+31.4 [+20.5, +41.9]", pct([p["did"], *ci["did"]])),
    ]
    a = b["report"]["all_backdoors_train"]
    rows.append(("  DiD all backdoors, 10 train carriers", "+1.4", pct([a["point"]["did"], *a["ci"]["did"]])))
    per = sorted(((i, v) for i, v in b["per_instance"].items() if v["catchable"]), key=lambda kv: -kv[1]["catchable"]["did"])
    rows.append(("  per-carrier DiD catchable (n)", "-",
                 ", ".join(f"{i} {v['catchable']['did'] * 100:+.0f} ({v['catchable_n']})" for i, v in per)))
    cp = b["clean_pass"]
    rows.append(("  clean pass naive S0/S1, receiver S0/S1", "-",
                 f"{rate(cp['naive_s0_pass'])}/{rate(cp['naive_s1_pass'])}, "
                 f"{rate(cp['receiver_s0_pass'])}/{rate(cp['receiver_s1_pass'])}"))
    rows.append(("  thresholds (cross-fitted, 5% FPR)", "-", ", ".join(f"{t:.2f}" for t in b["thresholds"])))

d = load("decoding")
if d:
    rows += [("2 Decoding: LOFO mean AUROC receiver", "0.94", ci3(d["mean_receiver"], "{:.3f}")),
             ("  LOFO mean AUROC naive", "0.67", ci3(d["mean_naive"], "{:.3f}"))]
    for F, v in d["per_family"].items():
        rows.append((f"  {F} (nested layer)", "-", f"L{v['layer']}: rec {v['receiver'][0]:.3f} / naive {v['naive'][0]:.3f}"))

m = load("mechanism")
if m:
    lw = m["late_window"]
    rows += [("3 Mechanism: receiver cos(d_S,d_B) late", "-0.65 (L53-61); < 0 from L47 (0.73)",
              f"{m['late_rec_cos']:+.3f} (L{lw[0]}-{lw[-1]}); negative from L{m['onset_rec_negative']}"
              f" ({m['onset_rel_depth']:.2f})" if m["onset_rec_negative"] is not None else
              f"{m['late_rec_cos']:+.3f} (L{lw[0]}-{lw[-1]}); never stays negative"),
             ("  naive cos late", "+0.53", f"{m['late_nai_cos']:+.3f}"),
             (f"  r(late cos, collusion), n={m['n_carriers']} carriers", "-0.89",
              ci3(m["r_late"], "{:+.2f}") + f" Fisher; boot [{m['r_late_bootstrap_ci'][0]:+.2f}, "
                                           f"{m['r_late_bootstrap_ci'][1]:+.2f}]")]

h = load("heads")
if h:
    e = h["effects"]
    rows += [("4 Heads: pairs (flip+aligned / used / problems)", "61 / 56 / -",
              f"{h['n_flip_aligned']} avail, {h['n_pairs']} run / {h['n_used']} used / {h['n_problems']}")]
    for n in (5, 10, 20, 40):
        rows.append((f"  top-{n} naive-B heads, share recovered", {5: "34% [27, 41]", 10: "61% [53, 68]", 20: "85% [80, 90]", 40: "96% [95, 98]"}[n],
                     f"{ci3(e[f'topB_{n}'], '{:+.0%}')} | layer-matched rand {e[f'randlm_mean_{n}'][0]:+.0%}"))
    best = max(h["layer_curve"].items(), key=lambda kv: kv[1][0])
    rows.append(("  top-20 layers (rel. depth min/med/max)", "L47-63 (0.73/0.80/0.98)",
                 f"{h['top20_layers']} ({'/'.join(f'{x:.2f}' for x in h['top20_rel_depth'])})"))
    rows.append(("  best whole-layer patch", "-", f"L{best[0]} {best[1][0]:+.0%}"))

h10 = load("heads_atp10")
if h10:
    e = h10["effects"]
    rows.append(("  [AtP over L10+] top-20 share / layer-matched", "-",
                 f"{ci3(e['topB_20'], '{:+.0%}')} / {e['randlm_mean_20'][0]:+.0%}; layers {h10['top20_layers'][0]}-"
                 f"{h10['top20_layers'][-1]} (median depth {h10['top20_rel_depth'][1]:.2f})"))

w = max(len(r[0]) for r in rows) if rows else 10
print(f"{'quantity':{w}s} | {'Qwen3-32B':24s} | {args.tag}")
print("-" * (w + 60))
for a, q, r in rows:
    print(f"{a:{w}s} | {q:24s} | {r}")
