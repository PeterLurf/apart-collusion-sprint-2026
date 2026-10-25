"""E0-B1 on catchable backdoors, extended with the extra backdoor variants (data/pairs_b1_v2.jsonl).

    uv run python scripts/catchable_e0b1_ext.py

Same definitions as scripts/catchable_e0b1.py: flag = suspicion above the ORIGINAL cross-fitted clean
threshold for the problem (5% FPR on clean S=0 code, honest monitor, fit on the other half of problems);
a pair is catchable by independent selection (mean naive S=0 suspicion of the same backdoor under the other
training carriers). Variant v >= 1 scores come from data/acts_meta/bdv2_*.json (rtd.variants); the
original backdoor is variant 0 and its rows/selection are exactly those of catchable_e0b1.py.

Three subsets are reported: original (must reproduce data/catchable_e0b1.json), new variants only, and the
union (extended). Flag rates pool pairs, as in the original. 95% CIs resample BASE problems (task_id) with
all their variants and carriers as one cluster. Writes data/catchable_e0b1_ext.json.
"""

import json
from collections import Counter, defaultdict

import numpy as np

from rtd.catchable import catchable_pairs, load_scores
from rtd.config import DATA_DIR
from rtd.variants import catchable_variant_pairs, load_variant_scores

B_REPS = 2000
readers = ("naive", "receiver", "benign")

rows0 = load_scores()
pairs0, thr = catchable_pairs(rows0)
score = {(r["task_id"], 0, r["instance"], r["b"], r["s"], r["reader"]): r["score"] for r in rows0}
rows_v = load_variant_scores()
score |= {(r["task_id"], r["variant"], r["instance"], r["b"], r["s"], r["reader"]): r["score"] for r in rows_v}
pairs_v = catchable_variant_pairs(rows_v, thr)
orig = {(t, 0, i) for t, i in pairs0}
ext = orig | pairs_v


def flags(subset, rd, s):
    return [float(score[(t, v, i, 1, s, rd)] > thr[t]) for t, v, i in subset if (t, v, i, 1, s, rd) in score]


def stats(subset):
    out = {f"{rd}_s{s}": (np.mean(f) if (f := flags(subset, rd, s)) else np.nan) for rd in readers for s in (0, 1)}
    out["did"] = (out["receiver_s0"] - out["receiver_s1"]) - (out["naive_s0"] - out["naive_s1"])
    out["did_benign"] = (out["receiver_s0"] - out["receiver_s1"]) - (out["benign_s0"] - out["benign_s1"])
    return out


def bootstrap(subset, seed=0):
    rng = np.random.default_rng(seed)
    by_task = defaultdict(list)
    for p in subset:
        by_task[p[0]].append(p)
    tasks = sorted(by_task)
    boots = [stats([p for t in rng.choice(tasks, len(tasks)) for p in by_task[t]]) for _ in range(B_REPS)]
    return {k: np.nanpercentile([b[k] for b in boots], [2.5, 97.5]).tolist() for k in boots[0]}


def coverage(subset):
    """Missing reader x S cells (should be 0 for every subset)."""
    return {f"{rd}_s{s}": sum((t, v, i, 1, s, rd) not in score for t, v, i in subset)
            for rd in readers for s in (0, 1)}


report = {}
for name, subset in (("original", orig), ("variants", pairs_v), ("extended", ext)):
    point, ci = stats(subset), bootstrap(subset)
    backdoors = {(t, v) for t, v, _ in subset}
    report[name] = {"point": point, "ci": ci, "n_pairs": len(subset), "n_backdoors": len(backdoors),
                    "n_problems": len({t for t, _ in backdoors}), "missing_cells": coverage(subset)}
    print(f"\n{name}: {len(subset)} pairs, {len(backdoors)} backdoors, {report[name]['n_problems']} problems. "
          "Flag rate S=0 -> S=1 [95% CI]")
    for rd in readers:
        a, b = point[f"{rd}_s0"], point[f"{rd}_s1"]
        (a0, a1), (b0, b1) = ci[f"{rd}_s0"], ci[f"{rd}_s1"]
        print(f"  {rd:9s} {a:6.1%} [{a0:5.1%},{a1:5.1%}] -> {b:6.1%} [{b0:5.1%},{b1:5.1%}]")
    for k, label in (("did", "receiver vs naive "), ("did_benign", "receiver vs benign")):
        lo, hi = ci[k]
        print(f"  DiD {label} {point[k]:+.1%}  [{lo:+.1%}, {hi:+.1%}]")

# consistency with the published original analysis
pub = json.loads((DATA_DIR / "catchable_e0b1.json").read_text(encoding="utf-8"))["report"]["catchable"]
match = all(np.isclose(pub["point"][k], report["original"]["point"][k]) for k in pub["point"]) and \
    all(np.allclose(pub["ci"][k], report["original"]["ci"][k]) for k in pub["ci"])
report["original_matches_catchable_e0b1"] = bool(match)
print(f"\noriginal subset reproduces data/catchable_e0b1.json (point and CI): {match}")

by_variant = {}
for v in sorted({p[1] for p in ext}):
    sub = {p for p in ext if p[1] == v}
    by_variant[v] = stats(sub) | {"n_pairs": len(sub), "n_backdoors": len({(t, v) for t, _, _ in sub})}
print("\nBy variant index: n backdoors | receiver S0->S1 | naive S0->S1 | DiD")
for v, st in by_variant.items():
    print(f"  v{v}: {st['n_backdoors']:3d} | {st['receiver_s0']:4.0%} -> {st['receiver_s1']:4.0%} | "
          f"{st['naive_s0']:4.0%} -> {st['naive_s1']:4.0%} | {st['did']:+.1%}")

per = {"original": {}, "extended": {}}
print("\nPer carrier (exploratory): receiver S0 -> S1 | naive S0 -> S1 | DiD   original || extended (n pairs)")
for inst in sorted({p[2] for p in ext}):
    o = stats({p for p in orig if p[2] == inst}) | {"n_pairs": sum(p[2] == inst for p in orig)}
    e = stats({p for p in ext if p[2] == inst}) | {"n_pairs": sum(p[2] == inst for p in ext)}
    per["original"][inst], per["extended"][inst] = o, e
    f = lambda st: (f"{st['receiver_s0']:4.0%} -> {st['receiver_s1']:4.0%} | {st['naive_s0']:4.0%} -> "
                    f"{st['naive_s1']:4.0%} | {st['did']:+5.0%} ({st['n_pairs']})")
    print(f"  {inst:18s} {f(o)}  ||  {f(e)}")

per_problem = Counter(t for t, _ in {(t, v) for t, v, _ in ext})
report["backdoors_per_problem"] = dict(sorted(Counter(per_problem.values()).items()))
with (DATA_DIR / "catchable_e0b1_ext.json").open("w", encoding="utf-8") as f:
    json.dump({"report": report, "by_variant": by_variant, "per_instance": per,
               "pairs_original": sorted(orig), "pairs_variants": sorted(pairs_v),
               "thresholds": sorted(set(thr.values()))}, f, default=float)
