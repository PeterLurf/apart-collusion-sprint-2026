"""E1 with nested layer selection and problem-level bootstrap CIs per held-out family.

    uv run python scripts/e1_nested.py

For each held-out family F, the layer is chosen from inner-fold scores only (probes that never saw F):
argmax over layers of the mean inner AUROC. The receiver's choice is also applied to the naive reader, so
both are compared at the same layer. Outer AUROCs come from per-row out-of-family logits
(data/nested_logits), resampled over problems for 95% CIs. Also checks the pre-registered criterion
(receiver >= 0.75 and naive <= 0.60 at the same layer). Writes data/e1_nested.json.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from rtd.config import DATA_DIR

B_REPS = 1000
inner = [json.loads(line) for line in (DATA_DIR / "nested_inner.jsonl").open(encoding="utf-8")]
logits = {}
for p in sorted((DATA_DIR / "nested_logits").glob("*.json")):
    logits[int(p.stem)] = json.loads(p.read_text(encoding="utf-8"))

score = defaultdict(list)
for r in inner:
    score[(r["reader"], r["outer"], r["layer"])].append(r["auroc"])
families = sorted({r["outer"] for r in inner})
rng = np.random.default_rng(0)


def auc_ci(rows):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append((r["s"], r["logit"]))
    tasks = sorted(by_task)
    flat = lambda ts: np.array([x for t in ts for x in by_task[t]])
    a = flat(tasks)
    point = roc_auc_score(a[:, 0], a[:, 1])
    boots = []
    for _ in range(B_REPS):
        b = flat(rng.choice(tasks, len(tasks)))
        if len(set(b[:, 0])) == 2:
            boots.append(roc_auc_score(b[:, 0], b[:, 1]))
    return point, *np.percentile(boots, [2.5, 97.5])


out, rec_all, nai_all = {}, [], []
print("held-out family | layer chosen (inner CV, receiver) | receiver AUROC [CI] | naive AUROC [CI] | gap")
for F in families:
    layer = max({k[2] for k in score if k[:2] == ("receiver", F)},
                key=lambda L: np.mean(score[("receiver", F, L)]))
    rows = [r for r in logits[layer] if r["family"] == F]
    rec = [r for r in rows if r["reader"] == "receiver"]
    nai = [r for r in rows if r["reader"] == "naive"]
    (rp, rl, rh), (n_p, nl, nh) = auc_ci(rec), auc_ci(nai)
    rec_all += [dict(r, logit=r["logit"]) for r in rec]
    nai_all += nai
    out[F] = {"layer": layer, "receiver": [rp, rl, rh], "naive": [n_p, nl, nh]}
    print(f"  {F:11s} L{layer:02d} | {rp:.3f} [{rl:.3f}, {rh:.3f}] | {n_p:.3f} [{nl:.3f}, {nh:.3f}] | {rp - n_p:+.3f}")

mean_rec = np.mean([v["receiver"][0] for v in out.values()])
mean_nai = np.mean([v["naive"][0] for v in out.values()])
print(f"\nmean over families (nested layer): receiver {mean_rec:.3f}  naive {mean_nai:.3f}  gap {mean_rec - mean_nai:+.3f}")

# pre-registered criterion at a single layer, chosen by nested CV on the pooled inner scores
pooled_layer = max({k[2] for k in score}, key=lambda L: np.mean(
    [a for (rd, F, LL), v in score.items() if rd == "receiver" and LL == L for a in v]))
fam_auc = lambda rd, L: np.mean([roc_auc_score([r["s"] for r in logits[L] if r["reader"] == rd and r["family"] == F],
                                               [r["logit"] for r in logits[L] if r["reader"] == rd and r["family"] == F])
                                 for F in families])
r_auc, n_auc = fam_auc("receiver", pooled_layer), fam_auc("naive", pooled_layer)
print(f"single layer chosen from inner scores: L{pooled_layer}  receiver {r_auc:.3f}  naive {n_auc:.3f}  "
      f"pre-registered (receiver >= 0.75 and naive <= 0.60): {'MET' if r_auc >= 0.75 and n_auc <= 0.60 else 'NOT MET'}")

with (DATA_DIR / "e1_nested.json").open("w", encoding="utf-8") as f:
    json.dump({"per_family": out, "mean_receiver": mean_rec, "mean_naive": mean_nai,
               "single_layer": pooled_layer, "single_layer_receiver": r_auc, "single_layer_naive": n_auc}, f,
              default=float)
