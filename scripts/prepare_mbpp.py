"""Select MBPP problems and write S=0/S=1 carrier pairs (B=0 only; backdoors come later).

    uv run python scripts/prepare_mbpp.py --n 300 --min-instances 12

Writes data/pairs.jsonl (one row per problem x instance) and prints applicability per instance.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from rtd.carriers import INSTANCES, make_pair
from rtd.config import DATA_DIR, N_PROBLEMS, SEED
from rtd.mbpp import Problem, load_mbpp, run_tests


def _pairs_for(p: Problem) -> dict:
    if not run_tests(p.code, p.tests, p.setup):
        return {}
    out = {}
    for inst in INSTANCES:
        pair = make_pair(inst, p.code, p.entry)
        if pair and run_tests(pair.s0, p.tests, p.setup) and run_tests(pair.s1, p.tests, p.setup):
            out[inst.id] = pair
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PROBLEMS)
    ap.add_argument("--min-instances", type=int, default=len(INSTANCES))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    problems = load_mbpp()
    print(f"loaded {len(problems)} MBPP problems with an entry function")
    with ThreadPoolExecutor(args.workers) as ex:
        all_pairs = list(ex.map(_pairs_for, problems))

    applicable = Counter(iid for pairs in all_pairs for iid in pairs)
    eligible = [(p, pairs) for p, pairs in zip(problems, all_pairs) if len(pairs) >= args.min_instances]
    random.Random(SEED).shuffle(eligible)
    chosen = eligible[: args.n]

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "pairs.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for p, pairs in chosen:
            for iid, pair in pairs.items():
                f.write(json.dumps({
                    "task_id": p.task_id, "text": p.text, "entry": p.entry,
                    "tests": p.tests, "setup": p.setup, "instance": iid,
                    "s0": pair.s0, "s1": pair.s1, "spans0": pair.spans0, "spans1": pair.spans1,
                }) + "\n")

    print("\napplicable (tests pass on both halves), out of all problems:")
    for inst in INSTANCES:
        print(f"  {inst.id:18s} {applicable[inst.id]:4d}")
    print(f"\neligible problems (>= {args.min_instances} instances): {len(eligible)}")
    print(f"wrote {len(chosen)} problems -> {out}")


if __name__ == "__main__":
    main()
