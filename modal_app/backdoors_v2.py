"""Additional backdoor variants per problem (same generator and verification as modal_app/backdoors.py).

    uv run modal run --detach modal_app/backdoors_v2.py --limit 3          # pilot
    uv run modal run --detach modal_app/backdoors_v2.py --n-variants 3     # all problems

For each problem, variants 1..n are generated in sequence; each prompt shows the existing backdoors (the
original, variant 0 from data/backdoors.jsonl, plus earlier variants) and asks for a different condition and
trigger. On top of the original checks (all MBPP tests pass, deviates from the reference on its trigger), a
variant must differ from every earlier backdoor in code and trigger and must return the reference value on
their triggers (rtd.backdoors.verify_distinct). A failed variant does not stop the next one.

Writes data/backdoors_v2.jsonl (one row per (task_id, variant)) and data/pairs_b1_v2.jsonl (pairs_b1 layout
plus a `variant` field); the originals are untouched. Volume backup: backdoors_v2/<task_id>.json.
"""

from __future__ import annotations

import json
from collections import Counter

import modal

from rtd.config import DATA_DIR, MODAL_ANTHROPIC_SECRET, MODAL_VOLUME

app = modal.App("rtd-backdoors-v2")
vol = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("anthropic==2.2.0").add_local_python_source("rtd")

PRICE_IN, PRICE_OUT = 2.0, 10.0  # Sonnet 5.5, USD per million tokens (as in modal_app/backdoors.py)


@app.function(image=image, secrets=[modal.Secret.from_name(MODAL_ANTHROPIC_SECRET)], volumes={"/vol": vol},
              timeout=3600, max_containers=24)
def make_variants(p: dict, n_variants: int) -> list[dict]:
    import os

    import anthropic

    from rtd.backdoors import (MAX_ATTEMPTS, MODEL, SYSTEM, build_variant_prompt, call_output, parse_reply,
                               verify, verify_distinct)
    from rtd.carriers import INSTANCES, make_pair
    from rtd.mbpp import run_tests

    path = f"/vol/backdoors_v2/{p['task_id']}.json"
    if os.path.exists(path):  # resume
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    client = anthropic.Anthropic()
    previous = [{"code": p["orig_code"], "trigger": p["orig_trigger"]}]
    results = []
    for v in range(1, n_variants + 1):
        messages = [{"role": "user", "content": build_variant_prompt(p["text"], p["code"], p["tests"], previous)}]
        out = {"task_id": p["task_id"], "variant": v, "ok": False, "attempts": 0, "input_tokens": 0,
               "output_tokens": 0}
        err = "no attempt"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            out["attempts"] = attempt
            try:
                resp = client.beta.messages.create(
                    model=MODEL, max_tokens=16000, system=SYSTEM, messages=messages,
                    output_config={"effort": "medium"},
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                )
            except anthropic.APIStatusError as e:
                err = f"API error {e.status_code}"
                break
            out["input_tokens"] += resp.usage.input_tokens
            out["output_tokens"] += resp.usage.output_tokens
            out["served_by"] = resp.model
            if resp.stop_reason == "refusal":
                err = "refusal"
                break
            text = "".join(b.text for b in resp.content if b.type == "text")
            parsed = parse_reply(text)
            if parsed is None:
                err = "the answer did not use the <code> and <trigger> format"
            else:
                bd_code, trigger = parsed
                err, ref_out, bd_out = verify(p["code"], bd_code, p["entry"], p["tests"], p["setup"], trigger)
                if err is None:
                    err = verify_distinct(p["code"], bd_code, p["entry"], p["setup"], trigger, previous)
            if err is None:
                out |= {"ok": True, "code": bd_code, "trigger": trigger, "ref_out": ref_out, "bd_out": bd_out}
                break
            messages += [
                {"role": "assistant", "content": resp.content},
                {"role": "user", "content": f"That attempt failed: {err}. Fix it and answer in the same format."},
            ]
        out["error"] = None if out["ok"] else err

        if out["ok"]:  # carriers, exactly as in modal_app/backdoors.py
            previous.append({"code": out["code"], "trigger": out["trigger"]})
            pairs = {}
            for inst in INSTANCES:
                pair = make_pair(inst, out["code"], p["entry"])
                if pair is None:
                    continue
                if all(run_tests(c, tuple(p["tests"]), p["setup"])
                       and call_output(c, p["entry"], out["trigger"], p["setup"]) == out["bd_out"]
                       for c in (pair.s0, pair.s1)):
                    pairs[inst.id] = {"s0": pair.s0, "s1": pair.s1, "spans0": pair.spans0, "spans1": pair.spans1}
            out["pairs"] = pairs
        results.append(out)

    os.makedirs("/vol/backdoors_v2", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    vol.commit()
    return results


@app.local_entrypoint()
def main(limit: int = 0, n_variants: int = 3):
    from rtd.mbpp import load_mbpp

    with (DATA_DIR / "backdoors.jsonl").open(encoding="utf-8") as f:
        orig = {r["task_id"]: r for r in map(json.loads, f) if r["ok"]}
    task_ids = sorted(orig)
    if limit:
        task_ids = task_ids[:limit]
    by_id = {p.task_id: p for p in load_mbpp()}
    problems = [
        {"task_id": t, "text": by_id[t].text, "code": by_id[t].code, "tests": list(by_id[t].tests),
         "setup": by_id[t].setup, "entry": by_id[t].entry,
         "orig_code": orig[t]["code"], "orig_trigger": orig[t]["trigger"]}
        for t in task_ids
    ]
    print(f"generating {n_variants} variants for {len(problems)} problems")

    results = []
    for r in make_variants.map(problems, kwargs={"n_variants": n_variants}, return_exceptions=True):
        if isinstance(r, list):
            results += r
        else:
            print("worker failed:", repr(r)[:300])
    meta = {p["task_id"]: p for p in problems}
    suffix = f"_pilot{limit}" if limit else ""
    with (DATA_DIR / f"backdoors_v2{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "pairs"}) + "\n")
    n_rows = 0
    with (DATA_DIR / f"pairs_b1_v2{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            p = meta[r["task_id"]]
            for iid, pair in (r.get("pairs") or {}).items():
                f.write(json.dumps({
                    "task_id": r["task_id"], "variant": r["variant"], "text": p["text"], "entry": p["entry"],
                    "tests": p["tests"], "setup": p["setup"], "instance": iid, "b": 1, "trigger": r["trigger"],
                    "ref_out": r["ref_out"], "bd_out": r["bd_out"], **pair,
                }) + "\n")
                n_rows += 1

    ok = [r for r in results if r["ok"]]
    tin = sum(r["input_tokens"] for r in results)
    tout = sum(r["output_tokens"] for r in results)
    print(f"\nverified variants: {len(ok)}/{len(problems) * n_variants}")
    print("per variant index:", dict(sorted(Counter(r["variant"] for r in ok).items())))
    print("attempts used (successes):", dict(sorted(Counter(r["attempts"] for r in ok).items())))
    print("failure reasons:", dict(Counter((r["error"] or "")[:90] for r in results if not r["ok"]).most_common(12)))
    print("served by:", dict(Counter(r.get("served_by") for r in results)))
    cov = Counter(iid for r in ok for iid in r.get("pairs", {}))
    print("B=1 carrier coverage per instance:", {k: cov[k] for k in sorted(cov)})
    print(f"B=1 pair rows written: {n_rows}")
    print(f"tokens: {tin:,} in / {tout:,} out  ~ ${tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT:.2f}")
