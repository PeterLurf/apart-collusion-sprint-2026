"""Extra backdoor variants per problem (data/pairs_b1_v2.jsonl): monitor jobs, scores, catchable selection.

Variant v >= 1 of problem t is a separate backdoor of the same problem; the original backdoor is variant 0.
Scores live in data/acts_meta/bdv2_*.json, which rtd.catchable.load_scores does not read, so every existing
analysis is unchanged. Catchability uses the ORIGINAL cross-fitted clean thresholds (rtd.catchable.
crossfit_thresholds on load_scores(): halves keyed by base task_id) and the same independent selection: pair
(t, v, i) is catchable when the honest monitor's mean S=0 suspicion of that variant under the other training
carriers exceeds the threshold for t.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from rtd.carriers import BY_ID, HARD_EVAL
from rtd.config import DATA_DIR
from rtd.jobs import job_row

PREFIX = "bdv2_"


def load_variant_pairs(path: Path = DATA_DIR / "pairs_b1_v2.jsonl") -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def variant_rows(pairs: list[dict], readers, s_values=(0, 1), keep=None) -> list[dict]:
    """Job rows (key-before placement) for training carriers; `keep(task_id, variant)` filters variants."""
    rows = []
    for p in pairs:
        inst = BY_ID[p["instance"]]
        if inst.id in HARD_EVAL or (keep is not None and not keep(p["task_id"], p["variant"])):
            continue
        for s in s_values:
            for reader in readers:
                rows.append(job_row(
                    f"{p['task_id']}/v{p['variant']}/{inst.id}/b1/s{s}/{reader}/before", p["text"], p[f"s{s}"],
                    p[f"spans{s}"], reader, inst.id, "before", task_id=p["task_id"], variant=p["variant"],
                    family=inst.family, b=1, s=s,
                ))
    return rows


def load_variant_scores(meta_dir: Path = DATA_DIR / "acts_meta") -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(meta_dir / f"{PREFIX}*.json"))):
        with open(p, encoding="utf-8") as fh:
            rows += json.load(fh)["rows"]
    rows = [r for r in rows if r.get("placement") == "before" and r["instance"] not in HARD_EVAL]
    dedup = {(r["task_id"], r["variant"], r["instance"], r["b"], r["s"], r["reader"]): r for r in rows}
    return list(dedup.values())


def catchable_variant_pairs(rows: list[dict], thr: dict[int, float]) -> set[tuple[int, int, str]]:
    s0 = {}
    for r in rows:
        if r["reader"] == "naive" and r["b"] == 1 and r["s"] == 0:
            s0.setdefault((r["task_id"], r["variant"]), {})[r["instance"]] = r["score"]
    pairs = set()
    for (t, v), by_inst in s0.items():
        for i in by_inst:
            others = [x for j, x in by_inst.items() if j != i]
            if others and np.mean(others) > thr[t]:
                pairs.add((t, v, i))
    return pairs
