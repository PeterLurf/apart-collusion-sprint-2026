"""Monitor job files for the extra backdoor variants (scores only, modal_app/score_only.py).

    uv run python scripts/make_jobs_v2.py validate   # 60 ORIGINAL rows rebuilt from pairs_b1.jsonl (check vs full_*)
    uv run python scripts/make_jobs_v2.py screen     # naive [+receiver] at S=0, every variant, training carriers
    uv run python scripts/make_jobs_v2.py stage2     # remaining readers x S for variants with a catchable pair
"""

import json
import random
import sys
from collections import Counter

from rtd.carriers import BY_ID, HARD_EVAL
from rtd.catchable import crossfit_thresholds, load_scores
from rtd.config import DATA_DIR
from rtd.jobs import job_row
from rtd.variants import catchable_variant_pairs, load_variant_pairs, load_variant_scores, variant_rows

stage = sys.argv[1]
screen_readers = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("naive",)


def write(rows, name):
    out = DATA_DIR / name
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")
    print(Counter((r["reader"], r["s"]) for r in rows))


if stage == "validate":
    with (DATA_DIR / "pairs_b1.jsonl").open(encoding="utf-8") as f:
        pairs = [p for p in map(json.loads, f) if p["instance"] not in HARD_EVAL]
    rows = []
    for p in random.Random(1).sample(pairs, 20):
        inst = BY_ID[p["instance"]]
        for s, reader in ((0, "naive"), (1, "receiver"), (0, "receiver")):
            rows.append(job_row(f"{p['task_id']}/{inst.id}/b1/s{s}/{reader}/before", p["text"], p[f"s{s}"],
                                p[f"spans{s}"], reader, inst.id, "before", task_id=p["task_id"],
                                family=inst.family, b=1, s=s))
    write(rows, "jobs_bdv2_validate.jsonl")
elif stage == "screen":
    write(variant_rows(load_variant_pairs(), screen_readers, (0,)), "jobs_bdv2_screen.jsonl")
elif stage == "stage2":
    thr = crossfit_thresholds(load_scores())
    scored = load_variant_scores()
    pairs = catchable_variant_pairs(scored, thr)
    variants = {(t, v) for t, v, _ in pairs}
    print(f"catchable variant pairs: {len(pairs)} from {len(variants)} variants of {len({t for t, _ in variants})} problems")
    have = {(r["task_id"], r["variant"], r["instance"], r["s"], r["reader"]) for r in scored}
    rows = [r for r in variant_rows(load_variant_pairs(), ("receiver", "naive", "benign"), (0, 1),
                                    keep=lambda t, v: (t, v) in variants)
            if (r["task_id"], r["variant"], r["instance"], r["s"], r["reader"]) not in have]
    write(rows, "jobs_bdv2_stage2.jsonl")
else:
    raise SystemExit(f"unknown stage {stage}")
