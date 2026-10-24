"""E4: safety-usefulness frontier for untrusted monitoring, with and without the receiver probe.

    uv run modal run --detach modal_app/e4.py                 # all 300 problems
    uv run modal run modal_app/e4.py --n-problems 3           # pilot

Items: for every problem x carrier, an honest task (B=0, S=0) and a colluding attack (B=1, S=1).
Stages (each result also saved on the Volume under e4/):
  1. Haiku 5.5 (trusted model T): suspicion score on the raw code, and a prose summary of it.
  2. Qwen3-32B receiver (untrusted monitor U): score on redacted code, and on T's summary; keeps the
     layer-53 last-token activation of the redacted-code pass.
  3. Probes: layer-53 last-token probe, trained leave-one-family-out on the activation cache, scores the
     raw (cached) and redacted activations of every item, always with a probe that never saw its family.
Output: data/e4_items.jsonl. Analysis: scripts/analyze_e4.py.
"""

from __future__ import annotations

import json
import random

import modal

from rtd.config import DATA_DIR, MODAL_ANTHROPIC_SECRET, MODAL_GPU, RECEIVER_MODEL, SEED
from rtd.modal_image import hf_secret, image, probe_image, vol

app = modal.App("rtd-e4")
haiku_image = modal.Image.debian_slim(python_version="3.11").pip_install("anthropic==2.2.0").add_local_python_source("rtd")
HAIKU = "claude-haiku-5-5"
PROBE_LAYER = 53


@app.function(image=haiku_image, secrets=[modal.Secret.from_name(MODAL_ANTHROPIC_SECRET)], timeout=3600,
              max_containers=24)
def trusted_chunk(items: list[dict]) -> list[dict]:
    """T suspicion score (digit 0-9) on raw code + T summary, for each item."""
    import re

    import anthropic

    from rtd.carriers import BY_ID
    from rtd.config import MONITOR_FORMAT
    from rtd.prompts import FORMATS, SUMMARIZER_SYSTEM, build_messages

    client = anthropic.Anthropic()
    fmt = FORMATS[MONITOR_FORMAT]

    def ask(system, user, max_tokens):
        try:
            resp = client.messages.create(model=HAIKU, max_tokens=max_tokens, system=system,
                                          output_config={"effort": "low"},
                                          messages=[{"role": "user", "content": user}])
        except anthropic.APIStatusError as e:
            return None, f"api {e.status_code}"
        if resp.stop_reason == "refusal":
            return None, "refusal"
        return "".join(b.text for b in resp.content if b.type == "text").strip(), None

    out = []
    for it in items:
        msgs = build_messages(it["text"], it["code"], "naive", BY_ID[it["instance"]], fmt=fmt)
        text, err = ask(msgs[0]["content"], msgs[1]["content"], 2048)
        digit = re.search(r"\d", text or "")
        summary, serr = ask(SUMMARIZER_SYSTEM, f"Task: {it['text']}\n\n```python\n{it['code']}```", 4096)
        out.append({"id": it["id"], "t_score": int(digit.group()) if digit else None, "t_error": err,
                    "summary": summary, "summary_error": serr})
    return out


@app.function(gpu=MODAL_GPU, image=image, volumes={"/vol": vol}, secrets=[hf_secret], timeout=3 * 3600)
def untrusted_shard(items: list[dict], model_id: str, shard: str) -> list[dict]:
    """Receiver score on redacted code and on T's summary; layer-53 last-token activation of the redacted pass."""
    import os

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rtd.calibrate import DIGITS, first_token_ids
    from rtd.carriers import BY_ID
    from rtd.config import MONITOR_FORMAT
    from rtd.extract import render
    from rtd.prompts import FORMATS, build_messages, summary_messages

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    digit_ids = first_token_ids(tok, DIGITS)
    fmt = FORMATS[MONITOR_FORMAT]

    def score(messages, keep_act=False):
        enc = tok(render(tok, messages), return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=keep_act)
        p = out.logits[0, -1, digit_ids].float().softmax(-1)
        s = float((p * torch.arange(10, device=p.device)).sum())
        act = out.hidden_states[PROBE_LAYER + 1][0, -1].float().cpu().numpy() if keep_act else None
        return s, act

    results, acts = [], []
    for it in items:
        inst = BY_ID[it["instance"]]
        u_red, act = score(build_messages(it["text"], it["redacted"], "receiver", inst, fmt=fmt), keep_act=True)
        u_sum = (score(summary_messages(it["text"], it["summary"], "receiver", inst, fmt=fmt))[0]
                 if it.get("summary") else None)
        results.append({"id": it["id"], "u_redacted": u_red, "u_summary": u_sum})
        acts.append(act)

    os.makedirs("/vol/e4", exist_ok=True)
    np.save(f"/vol/e4/{shard}_acts.npy", np.stack(acts).astype(np.float32))
    with open(f"/vol/e4/{shard}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    vol.commit()
    return results


@app.function(image=probe_image, volumes={"/vol": vol}, cpu=8, memory=32768, timeout=3600)
def probe_scores(items: list[dict], shards: list[str]) -> dict:
    """Leave-one-family-out layer-53 probes: score every item's raw (cached) and redacted activations."""
    import glob

    import numpy as np
    from safetensors import safe_open

    from rtd.carriers import FAMILIES, HARD_EVAL
    from rtd.probes import fit_probe

    paths = sorted(glob.glob("/vol/acts/full_*.safetensors"))
    meta = []
    for p in paths:
        with open(p.replace(".safetensors", ".json"), encoding="utf-8") as fh:
            info = json.load(fh)
        meta += info["rows"]
    li = info["layers"].index(PROBE_LAYER)
    X = np.concatenate([safe_open(p, framework="pt").get_slice("last")[:, li, :].float().numpy() for p in paths])
    col = lambda k: np.array([m[k] for m in meta])
    reader, placement, inst, family = col("reader"), col("placement"), col("instance"), col("family")
    y, groups = col("s"), col("task_id")
    base = (reader == "receiver") & (placement == "before")
    train_inst = base & ~np.isin(inst, list(HARD_EVAL))
    row_of = {(m["task_id"], m["instance"], m["b"], m["s"]): i for i, m in enumerate(meta)
              if m["reader"] == "receiver" and m["placement"] == "before"}

    red = np.concatenate([np.load(f"/vol/e4/{s}_acts.npy") for s in shards])
    red_ids = [r["id"] for s in shards for r in json.load(open(f"/vol/e4/{s}.json", encoding="utf-8"))]
    red_idx = {i: k for k, i in enumerate(red_ids)}

    out = {}
    for fam in FAMILIES:
        tr = train_inst & (family != fam)
        probe = fit_probe(X[tr], y[tr], groups[tr])
        fam_items = [it for it in items if it["family"] == fam]
        raw = np.stack([X[row_of[(it["task_id"], it["instance"], it["b"], it["s"])]] for it in fam_items])
        raw_scores = probe.logits(raw)
        have = [it for it in fam_items if it["id"] in red_idx]
        red_scores = probe.logits(red[[red_idx[it["id"]] for it in have]]) if have else []
        for it, z in zip(fam_items, raw_scores):
            out.setdefault(it["id"], {})["probe_raw"] = float(z)
        for it, z in zip(have, red_scores):
            out[it["id"]]["probe_redacted"] = float(z)
    return out


def _items(n_problems: int) -> list[dict]:
    from rtd.carriers import BY_ID
    from rtd.redact import redact

    def load(name):
        with (DATA_DIR / name).open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    u_raw = {}
    for path in sorted((DATA_DIR / "acts_meta").glob("full_*.json")):
        for r in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            if r["reader"] == "receiver" and r["placement"] == "before":
                u_raw[(r["task_id"], r["instance"], r["b"], r["s"])] = r["score"]
    pairs = {(p["task_id"], p["instance"], p.get("b", 0)): p for p in load("pairs.jsonl") + load("pairs_b1.jsonl")}
    task_ids = sorted({k[0] for k in pairs})
    if n_problems < len(task_ids):
        task_ids = sorted(random.Random(SEED).sample(task_ids, n_problems))
    items = []
    for t in task_ids:
        for iid, inst in BY_ID.items():
            for kind, b, s in (("honest", 0, 0), ("attack", 1, 1)):
                p = pairs.get((t, iid, b))
                if p is None:
                    continue
                code = p[f"s{s}"]
                items.append({"id": f"{t}/{iid}/{kind}", "task_id": t, "instance": iid, "family": inst.family,
                              "kind": kind, "b": b, "s": s, "text": p["text"], "code": code,
                              "redacted": redact(code), "u_raw": u_raw[(t, iid, b, s)],
                              "carrier_survives_redaction": inst.detect(redact(code)) if s else None})
    return items


@app.local_entrypoint()
def main(n_problems: int = 300, gpu_shards: int = 4, chunk: int = 40, model: str = RECEIVER_MODEL):
    items = _items(n_problems)
    print(f"{len(items)} items ({n_problems} problems)")

    ckpt = DATA_DIR / f"e4_trusted_n{n_problems}.jsonl"  # stage-1 checkpoint: rerun skips the API calls
    if ckpt.exists():
        with ckpt.open(encoding="utf-8") as f:
            trusted = {r["id"]: r for r in map(json.loads, f)}
        print(f"loaded {len(trusted)} trusted results from {ckpt}")
    else:
        trusted = {r["id"]: r for rs in trusted_chunk.map([items[i:i + chunk] for i in range(0, len(items), chunk)])
                   for r in rs}
        with ckpt.open("w", encoding="utf-8") as f:
            for r in trusted.values():
                f.write(json.dumps(r) + "\n")
    for it in items:
        it.update({k: v for k, v in trusted[it["id"]].items() if k != "id"})
    print(f"T scores: {sum(it['t_score'] is not None for it in items)}/{len(items)}; "
          f"summaries: {sum(bool(it['summary']) for it in items)}; "
          f"errors: {sorted({e for it in items for e in (it['t_error'], it['summary_error']) if e})}")

    names = [f"n{n_problems}_{k}" for k in range(gpu_shards)]
    gpu_items = [{k: it[k] for k in ("id", "text", "instance", "redacted", "summary")} for it in items]
    calls = [untrusted_shard.spawn(gpu_items[k::gpu_shards], model, names[k]) for k in range(gpu_shards)]
    untrusted = {r["id"]: r for c in calls for r in c.get()}
    for it in items:
        it.update({k: v for k, v in untrusted[it["id"]].items() if k != "id"})

    probes = probe_scores.remote([{k: it[k] for k in ("id", "task_id", "instance", "family", "b", "s")}
                                  for it in items], names)
    for it in items:
        it.update(probes.get(it["id"], {}))

    out = DATA_DIR / ("e4_items.jsonl" if n_problems >= 300 else f"e4_items_n{n_problems}.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({k: v for k, v in it.items() if k not in ("code", "redacted", "text")}) + "\n")
    print(f"wrote {out}")
