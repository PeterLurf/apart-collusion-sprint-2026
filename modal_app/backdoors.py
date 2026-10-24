"""Generate test-verified backdoors (B=1) with Claude Sonnet 5.5 on Modal, then apply the carriers.

    uv run modal run --detach modal_app/backdoors.py --limit 5        # pilot
    uv run modal run --detach modal_app/backdoors.py                  # all problems in data/pairs.jsonl

Writes data/backdoors.jsonl (one row per problem) and data/pairs_b1.jsonl (same layout as
data/pairs.jsonl, plus b=1 and trigger fields). Each result is also saved on the Volume under
backdoors/<task_id>.json, so a dropped client loses nothing: fetch with
    uv run modal volume get rtd-cache backdoors/ data/backdoors_vol/
Refusals fall back server-side to another model (fallbacks="default").
"""

from __future__ import annotations

import json
from collections import Counter

import modal

from rtd.config import DATA_DIR, MODAL_ANTHROPIC_SECRET, MODAL_VOLUME

app = modal.App("rtd-backdoors")
vol = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("anthropic==2.2.0").add_local_python_source("rtd")

PRICE_IN, PRICE_OUT = 2.0, 10.0  # Sonnet 5.5, USD per million tokens


@app.function(image=image, secrets=[modal.Secret.from_name(MODAL_ANTHROPIC_SECRET)], volumes={"/vol": vol},
              timeout=1200, max_containers=16)
def make_backdoor(p: dict) -> dict:
    import os

    import anthropic

    from rtd.backdoors import MAX_ATTEMPTS, MODEL, SYSTEM, build_prompt, call_output, parse_reply, verify
    from rtd.carriers import INSTANCES, make_pair
    from rtd.mbpp import run_tests

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": build_prompt(p["text"], p["code"], p["tests"])}]
    out = {"task_id": p["task_id"], "ok": False, "attempts": 0, "input_tokens": 0, "output_tokens": 0}
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
            ref_out = bd_out = None
        else:
            bd_code, trigger = parsed
            err, ref_out, bd_out = verify(p["code"], bd_code, p["entry"], p["tests"], p["setup"], trigger)
        if err is None:
            out |= {"ok": True, "code": bd_code, "trigger": trigger, "ref_out": ref_out, "bd_out": bd_out}
            break
        messages += [
            {"role": "assistant", "content": resp.content},
            {"role": "user", "content": f"That attempt failed: {err}. Fix it and answer in the same format."},
        ]
    out["error"] = None if out["ok"] else err

    if out["ok"]:  # carriers on the backdoored code; both halves must keep the tests and the backdoor
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

    os.makedirs("/vol/backdoors", exist_ok=True)
    with open(f"/vol/backdoors/{p['task_id']}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    vol.commit()
    return out


@app.local_entrypoint()
def main(limit: int = 0):
    from rtd.mbpp import load_mbpp

    with (DATA_DIR / "pairs.jsonl").open(encoding="utf-8") as f:
        task_ids = sorted({json.loads(line)["task_id"] for line in f})
    if limit:
        task_ids = task_ids[:limit]
    by_id = {p.task_id: p for p in load_mbpp()}
    problems = [
        {"task_id": t, "text": by_id[t].text, "code": by_id[t].code, "tests": list(by_id[t].tests),
         "setup": by_id[t].setup, "entry": by_id[t].entry}
        for t in task_ids
    ]
    print(f"generating backdoors for {len(problems)} problems")

    results = [r for r in make_backdoor.map(problems, return_exceptions=True) if isinstance(r, dict)]
    meta = {p["task_id"]: p for p in problems}
    suffix = f"_pilot{limit}" if limit else ""
    with (DATA_DIR / f"backdoors{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "pairs"}) + "\n")
    n_rows = 0
    with (DATA_DIR / f"pairs_b1{suffix}.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            p = meta[r["task_id"]]
            for iid, pair in (r.get("pairs") or {}).items():
                f.write(json.dumps({
                    "task_id": r["task_id"], "text": p["text"], "entry": p["entry"], "tests": p["tests"],
                    "setup": p["setup"], "instance": iid, "b": 1, "trigger": r["trigger"],
                    "ref_out": r["ref_out"], "bd_out": r["bd_out"], **pair,
                }) + "\n")
                n_rows += 1

    ok = [r for r in results if r["ok"]]
    tin = sum(r["input_tokens"] for r in results)
    tout = sum(r["output_tokens"] for r in results)
    print(f"\nverified backdoors: {len(ok)}/{len(problems)}  (crashed workers: {len(problems) - len(results)})")
    print("attempts used (successes):", dict(sorted(Counter(r["attempts"] for r in ok).items())))
    print("failure reasons:", dict(Counter(r["error"] for r in results if not r["ok"]).most_common()))
    print("served by:", dict(Counter(r.get("served_by") for r in results)))
    cov = Counter(iid for r in ok for iid in r.get("pairs", {}))
    print("B=1 carrier coverage per instance:", {k: cov[k] for k in sorted(cov)})
    print(f"B=1 pair rows written: {n_rows}")
    print(f"tokens: {tin:,} in / {tout:,} out  ≈ ${tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT:.2f}")
