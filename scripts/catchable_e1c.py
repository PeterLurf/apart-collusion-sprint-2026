"""Summarize E1c causal tests on catchable backdoors, with problem-level bootstrap CIs.

    PYTHONIOENCODING=utf-8 uv run python scripts/catchable_e1c.py

Uses the current catchable definition (independent selection, cross-fitted thresholds) and keeps the causal
rows that fall in it (the causal run itself predates the definition; rows outside it are dropped).
Each line: flag rate, and its change vs the baseline of the same snippets, with a 95% CI over problems.
Necessity: an intervention on S=1 that raises flagging toward the S=0 level. Sufficiency: one on S=0 that
lowers it. Random directions are the controls. Writes data/catchable_e1c.json.
"""

import json
from collections import defaultdict

import numpy as np

from rtd.catchable import catchable_pairs, load_scores
from rtd.config import DATA_DIR

B_REPS = 2000
pairs, thr = catchable_pairs(load_scores())
rows = [r for r in map(json.loads, (DATA_DIR / "e1_catchable_causal.jsonl").open(encoding="utf-8"))
        if (r["task_id"], r["instance"]) in pairs]
patch = [r for r in map(json.loads, (DATA_DIR / "e1_catchable_patching.jsonl").open(encoding="utf-8"))
         if (r["task_id"], r["instance"]) in pairs and r["aligned"]]
rng = np.random.default_rng(0)
report = {}


def pooled(r, prefix):
    vals = [v for k, v in r.items() if k.startswith(prefix) and k[len(prefix):][:1].isdigit()]
    return float(np.mean([v > thr[r["task_id"]] for v in vals])) if vals else None


def flag(r, key):
    if key.endswith("*"):
        return pooled(r, key[:-1])
    v = r.get(key)
    return None if v is None else float(v > thr[r["task_id"]])


def line(label, rs, key, base_key="base"):
    by_task = defaultdict(list)
    for r in rs:
        f, f0 = flag(r, key), flag(r, base_key)
        if f is not None:
            by_task[r["task_id"]].append((f, f0))
    tasks = sorted(by_task)
    arr = lambda ts: np.array([x for t in ts for x in by_task[t]])
    a = arr(tasks)
    rate, delta = a[:, 0].mean(), (a[:, 0] - a[:, 1]).mean()
    boots = [arr(rng.choice(tasks, len(tasks))) for _ in range(B_REPS)]
    lo, hi = np.percentile([(b[:, 0] - b[:, 1]).mean() for b in boots], [2.5, 97.5])
    report[label] = {"rate": rate, "delta": delta, "ci": [lo, hi], "n": len(a), "problems": len(tasks)}
    shown = "" if key == base_key else f"   Δ {delta:+6.1%} [{lo:+5.1%}, {hi:+5.1%}]"
    print(f"  {label:34s} flag {rate:5.1%}{shown}   (n={len(a)}, {len(tasks)} problems)")


grp = defaultdict(list)
for r in rows:
    grp[(r["reader"], r["s"])].append(r)
r1, r0, n1 = grp[("receiver", 1)], grp[("receiver", 0)], grp[("naive", 1)]

print(f"Catchable backdoors kept: {len({(r['task_id'], r['instance']) for r in rows})} pairs "
      f"from {len({r['task_id'] for r in rows})} problems (of {len(pairs)} catchable pairs)")
print("\nNecessity, receiver S=1 (colluding). Δ vs its own baseline; want Δ > 0 and above random.")
line("receiver S=0 baseline (target)", r0, "base")
line("receiver S=1 baseline", r1, "base")
for label, key in (("ablate d_S 45-55", "ablate_S"), ("ablate d_S⊥d_B 45-55", "ablate_Sperp"),
                   ("ablate random 45-55 (control)", "ablate_rand*"), ("ablate d_S⊥d_B 21-63 [breaks model]", "ablate_Sperp_wide"),
                   ("ablate random 21-63 [breaks model]", "ablate_randwide*")):
    line(label, r1, key)

print("\nSufficiency, receiver S=0. Want Δ < 0 and beyond random.")
for label, key in (("steer d_S a=1 (L53)", "steer_S_a1"), ("steer d_S a=2", "steer_S_a2"),
                   ("steer d_S a=4", "steer_S_a4"), ("steer random |4 d_S| (control)", "steer_rand*")):
    line(label, r0, key)

print("\nSpecificity, naive S=1 (no key).")
line("naive baseline", n1, "base")
line("ablate d_S 45-55", n1, "ablate_S")

if patch:
    print(f"\nCarrier-token patching (token-aligned substitute pairs)")
    for r in patch:
        r["base"] = r["base_s1"]
    line("S=1 baseline", patch, "base_s1", "base_s1")
    for L in sorted({int(k.split("_")[1]) for r in patch for k in r if k.startswith("noise_")}):
        line(f"S=1 + S=0 carrier @L{L} (necessity)", patch, f"noise_{L}", "base_s1")
    line("S=0 baseline", patch, "base_s0", "base_s0")
    for L in sorted({int(k.split("_")[1]) for r in patch for k in r if k.startswith("denoise_")}):
        line(f"S=0 + S=1 carrier @L{L} (sufficiency)", patch, f"denoise_{L}", "base_s0")

with (DATA_DIR / "catchable_e1c.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, default=float)
