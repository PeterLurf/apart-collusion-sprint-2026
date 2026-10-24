"""Insert vs substitute split and code-length baseline for the E1 probes.

    uv run python scripts/confound_split.py [layer]

Insert carriers make S=1 code longer, so "an extra statement is present" could be a family-general cue.
  1. Probe AUROC (out-of-family logits, data/nested_logits) by carrier kind, receiver vs naive.
  2. Length-only baseline: AUROC of S from code length (chars, lines), and a leave-one-family-out
     logistic probe on [chars, lines, tokens-ish] - the best a pure length detector does on a new family.
95% CIs resample problems.
"""

import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from rtd.carriers import BY_ID, HARD_EVAL
from rtd.config import DATA_DIR

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 53
rng = np.random.default_rng(0)


def auc_ci(rows, key, B=1000):
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append((r["s"], r[key]))
    tasks = sorted(by_task)
    flat = lambda ts: np.array([x for t in ts for x in by_task[t]])
    a = flat(tasks)
    boots = [roc_auc_score(b[:, 0], b[:, 1]) for b in (flat(rng.choice(tasks, len(tasks))) for _ in range(B))
             if len(set(b[:, 0])) == 2]
    return roc_auc_score(a[:, 0], a[:, 1]), *np.percentile(boots, [2.5, 97.5])


fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
logits = json.loads((DATA_DIR / "nested_logits" / f"{LAYER:02d}.json").read_text(encoding="utf-8"))
print(f"1) Probe AUROC by carrier kind, layer {LAYER}, out-of-family logits")
for kind in ("substitute", "insert"):
    for rd in ("receiver", "naive"):
        rows = [r for r in logits if r["reader"] == rd and BY_ID[r["instance"]].kind == kind]
        print(f"   {kind:10s} {rd:9s} {fmt(auc_ci(rows, 'logit'))}   ({len({r['instance'] for r in rows})} carriers)")

pairs = []
for name in ("pairs.jsonl", "pairs_b1.jsonl"):
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        pairs += [json.loads(line) for line in f]
rows = []
for p in pairs:
    if p["instance"] in HARD_EVAL:
        continue
    for s in (0, 1):
        code = p[f"s{s}"]
        rows.append({"task_id": p["task_id"], "instance": p["instance"], "family": BY_ID[p["instance"]].family,
                     "kind": BY_ID[p["instance"]].kind, "s": s, "chars": len(code),
                     "lines": code.count("\n"), "toks": len(code.split())})

print("\n2) Length-only baseline (S from code length)")
for kind in ("substitute", "insert", "all"):
    sub = [r for r in rows if kind == "all" or r["kind"] == kind]
    print(f"   {kind:10s} chars {fmt(auc_ci(sub, 'chars'))}   lines {fmt(auc_ci(sub, 'lines'))}")

feats = lambda rs: np.array([[r["chars"], r["lines"], r["toks"]] for r in rs], float)
print("   leave-one-family-out logistic probe on [chars, lines, tokens], per held-out family:")
for F in sorted({r["family"] for r in rows}):
    tr = [r for r in rows if r["family"] != F]
    te = [r for r in rows if r["family"] == F]
    mu, sd = feats(tr).mean(0), feats(tr).std(0) + 1e-9
    clf = LogisticRegression(max_iter=1000).fit((feats(tr) - mu) / sd, [r["s"] for r in tr])
    for r, z in zip(te, clf.decision_function((feats(te) - mu) / sd)):
        r["len_logit"] = z
    print(f"     {F:11s} {fmt(auc_ci(te, 'len_logit'))}")
