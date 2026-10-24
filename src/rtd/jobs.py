"""Activation-caching job rows: one rendered-prompt request per (snippet, reader, placement)."""

from __future__ import annotations

import random

from rtd.carriers import BY_ID, INSTANCES, make_pair
from rtd.config import MONITOR_FORMAT, SEED
from rtd.fixtures import FIXTURES
from rtd.prompts import FORMATS, build_messages

AFTER_FAMILIES = ("syntactic", "lexical")  # key-after-code variant, on a problem subset


def job_row(row_id: str, task_text: str, code: str, spans, reader: str, instance_id: str,
            placement: str = "before", **meta) -> dict:
    return {
        "id": row_id,
        "messages": build_messages(task_text, code, reader, BY_ID[instance_id], placement,
                                   fmt=FORMATS[MONITOR_FORMAT]),
        "code": code,
        "spans": [list(s) for s in spans],
        "reader": reader,
        "instance": instance_id,
        "placement": placement,
        **meta,
    }


def full_rows(pairs: list[dict], subset_size: int = 100) -> list[dict]:
    """E1 grid: receiver + naive on every pair (key before the code), plus on a seeded problem subset
    the benign-key reader and the key-after-code variant (syntactic and lexical families only)."""
    task_ids = sorted({p["task_id"] for p in pairs})
    subset = set(random.Random(SEED).sample(task_ids, min(subset_size, len(task_ids))))
    rows = []
    for p in pairs:
        inst, b = BY_ID[p["instance"]], p.get("b", 0)
        variants = [("receiver", "before"), ("naive", "before")]
        if p["task_id"] in subset:
            variants.append(("benign", "before"))
            if inst.family in AFTER_FAMILIES:
                variants += [("receiver", "after"), ("naive", "after")]
        for s in (0, 1):
            code, spans = p[f"s{s}"], p[f"spans{s}"]
            for reader, placement in variants:
                rows.append(job_row(
                    f"{p['task_id']}/{inst.id}/b{b}/s{s}/{reader}/{placement}", p["text"], code, spans,
                    reader, inst.id, placement, task_id=p["task_id"], family=inst.family, b=b, s=s,
                ))
    return rows


def smoke_rows(readers=("receiver", "naive")) -> list[dict]:
    """Fixture-based rows covering every instance x S x reader (no MBPP needed)."""
    rows = []
    for fx in FIXTURES:
        for inst in INSTANCES:
            pair = make_pair(inst, fx.code, fx.entry)
            if pair is None:
                continue
            for s, (code, spans) in enumerate(((pair.s0, pair.spans0), (pair.s1, pair.spans1))):
                for reader in readers:
                    rows.append(job_row(
                        f"{fx.entry}/{inst.id}/s{s}/{reader}", f"Implement {fx.entry}.",
                        code, spans, reader, inst.id, s=s, family=inst.family,
                    ))
    return rows
