"""Write data/jobs_full.jsonl: the E1 caching grid over B=0 and B=1 pairs.

    uv run python scripts/make_jobs.py
"""

import json
from collections import Counter

from rtd.config import DATA_DIR
from rtd.jobs import full_rows


def load(name):
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


rows = full_rows(load("pairs.jsonl") + load("pairs_b1.jsonl"))
out = DATA_DIR / "jobs_full.jsonl"
with out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows -> {out}")
print(Counter((r["reader"], r["placement"]) for r in rows))
