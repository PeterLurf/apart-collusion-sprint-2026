"""False-positive cost of the d_S-perp-d_B ablation (audit item 9).

    uv run python scripts/ablation_fpcost.py

Reads data/e1_fpcost.jsonl from `modal run modal_app/causal.py::fpcost`: the same ablations as the catchable causal
run, applied to the CLEAN code of the catchable problems (receiver without and with the carrier, naive reader).
Flags use the same cross-fitted threshold as the attacks. Reports each ablation's flag rate and its change from
the unablated baseline, with 95% CIs over problems (1,000 resamples). Writes data/ablation_fpcost.json.
"""

import json
import random
from collections import defaultdict

import numpy as np

from rtd.catchable import catchable_pairs, load_scores
from rtd.config import DATA_DIR

CONDS = ("ablate_Sperp", "ablate_Sperp_wide", "ablate_rand0", "ablate_randwide0")
LABEL = {"ablate_Sperp": "S-perp, layers 45-55", "ablate_Sperp_wide": "S-perp, layers 21-63",
         "ablate_rand0": "random, layers 45-55", "ablate_randwide0": "random, layers 21-63"}

_, thr = catchable_pairs(load_scores())
fp = [json.loads(x) for x in (DATA_DIR / "e1_fpcost.jsonl").open(encoding="utf-8")]
flag = lambda g, c: float(np.mean([r[c] > thr[r["task_id"]] for r in g]))
out = {}
for rd, s in (("receiver", 0), ("receiver", 1), ("naive", 0)):
    g = [r for r in fp if r["reader"] == rd and r["s"] == s]
    by_task = defaultdict(list)
    for r in g:
        by_task[r["task_id"]].append(r)
    tasks, rng = sorted(by_task), random.Random(0)
    cell = {"n": len(g), "n_problems": len(tasks), "base": flag(g, "base")}
    print(f"{rd} clean code, S={s} (n={len(g)}): unablated {cell['base']:.1%}")
    for c in CONDS:
        boots = []
        for _ in range(1000):
            smp = [r for t in rng.choices(tasks, k=len(tasks)) for r in by_task[t]]
            boots.append(flag(smp, c) - flag(smp, "base"))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cell[c] = {"flag": flag(g, c), "change": flag(g, c) - cell["base"], "ci": [float(lo), float(hi)]}
        print(f"  {LABEL[c]:22s} {cell[c]['flag']:6.1%}  change {100 * cell[c]['change']:+5.1f} "
              f"[{100 * lo:+.1f}, {100 * hi:+.1f}]")
    out[f"{rd}_s{s}"] = cell
(DATA_DIR / "ablation_fpcost.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote data/ablation_fpcost.json")
