"""Problem-resampled 95% CIs for the M4 patching effects in data/m4_circuit.json.

    uv run python scripts/m4_bootstrap.py

Same effect definition as modal_app/circuit.py (share of the contrast recovered, pairs with |base gap| > 0.5),
resampling problems (task ids) with replacement. Output: data/m4_bootstrap.json
"""

import json
import random

import numpy as np

from rtd.config import DATA_DIR, SEED

REPS = 1000
d = json.loads((DATA_DIR / "m4_circuit.json").read_text(encoding="utf-8"))
ver = d["verify"]
by_task = {}
for v in ver:
    by_task.setdefault(v["key"].split("/")[0], []).append(v)
tasks = sorted(by_task)


def effect(rows, name, prefix=""):
    vals = [(v[prefix + name] - v[prefix + "base_clean"]) / (v[prefix + "base_corrupt"] - v[prefix + "base_clean"])
            for v in rows if abs(v[prefix + "base_corrupt"] - v[prefix + "base_clean"]) > 0.5]
    return float(np.mean(vals)) if vals else float("nan")


names = [(n, "") for n in d["effects"]] + [(n, "naive_") for n in d["naive_effects"]]
rng = random.Random(SEED)
samples = {p + n: [] for n, p in names}
for _ in range(REPS):
    rows = [v for t in rng.choices(tasks, k=len(tasks)) for v in by_task[t]]
    for n, p in names:
        samples[p + n].append(effect(rows, n, p))
out = {"n_pairs": len(ver), "n_tasks": len(tasks), "reps": REPS,
       "ci": {k: [effect(ver, k.removeprefix("naive_"), "naive_" if k.startswith("naive_") else ""),
                  *map(float, np.nanpercentile(s, [2.5, 97.5]))] for k, s in samples.items()}}
(DATA_DIR / "m4_bootstrap.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for k in ["topB_5", "topB_10", "topB_20", "topB_40", "layer_47", "layer_49", "layer_50", "layer_53"]:
    for p in ("", "naive_"):
        if p + k in out["ci"]:
            e, lo, hi = out["ci"][p + k]
            print(f"{p + k:18s} {e:+.0%} [{lo:+.0%}, {hi:+.0%}]")
print(f"wrote data/m4_bootstrap.json ({len(ver)} pairs, {len(tasks)} problems)")
