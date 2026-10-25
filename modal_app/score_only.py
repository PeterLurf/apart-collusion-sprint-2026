"""Monitor suspicion scores only (no activations): the same forward pass and score as cache_activations.py.

    uv run modal run modal_app/score_only.py --jobs data/jobs_bdv2_screen.jsonl --out bdv2_screen

Per row: render the chat prompt exactly as rtd.extract.extract_row does, one unbatched forward pass, score =
expected suspicion digit (softmax over the logits of "0".."9" at the last prompt token). Nothing is cached
but the scores. Each shard's meta (same layout as data/acts_meta/full_*.json, minus layers/activations) is
saved on the Volume under scores/<out>_<shard>.json (resume) and locally as data/acts_meta/<out>_<shard>.json.
"""

from __future__ import annotations

import json
import time

import modal

from rtd.config import DATA_DIR, MODAL_GPU, RECEIVER_MODEL
from rtd.modal_image import hf_secret, image, vol

app = modal.App("rtd-score-only")


@app.function(gpu=MODAL_GPU, image=image, volumes={"/vol": vol}, secrets=[hf_secret], timeout=3 * 3600,
              max_containers=8)
def score_shard(rows: list[dict], model_id: str, out_name: str) -> dict:
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rtd.calibrate import DIGITS, first_token_ids
    from rtd.extract import render

    path = f"/vol/scores/{out_name}.json"
    if os.path.exists(path):  # resume: shard already scored
        with open(path, encoding="utf-8") as fh:
            return json.load(fh) | {"skipped": True}

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    digit_ids = first_token_ids(tok, DIGITS)
    t_load = time.time() - t0

    meta = []
    for row in rows:
        enc = tok(render(tok, row["messages"]), add_special_tokens=False, return_tensors="pt")
        with torch.inference_mode():
            out = model(**{k: v.to(model.device) for k, v in enc.items()})
        p = out.logits[0, -1, digit_ids].float().softmax(-1)
        score = (p * torch.arange(10, device=p.device)).sum().item()
        meta.append({k: v for k, v in row.items() if k != "messages"} | {"score": score})

    result = {"model": model_id, "rows": meta, "load_s": t_load, "score_s": time.time() - t0 - t_load}
    os.makedirs("/vol/scores", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    vol.commit()
    return result


@app.local_entrypoint()
def main(jobs: str, out: str, model: str = RECEIVER_MODEL, shard_size: int = 1500):
    with open(jobs, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    shards = [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]
    names = [f"{out}_{k:03d}" for k in range(len(shards))]
    print(f"{len(rows)} rows in {len(shards)} shards")
    gpu_s = 0.0
    for name, res in zip(names, score_shard.starmap([(s, model, n) for s, n in zip(shards, names)])):
        with (DATA_DIR / "acts_meta" / f"{name}.json").open("w", encoding="utf-8") as fh:
            json.dump({"model": res["model"], "rows": res["rows"]}, fh)
        gpu_s += 0 if res.get("skipped") else res["load_s"] + res["score_s"]
        print(f"{name}: {len(res['rows'])} rows, load {res['load_s']:.0f}s, score {res['score_s']:.0f}s"
              f"{' (cached)' if res.get('skipped') else ''}")
    print(f"GPU seconds this run (load + score, excl. container start): {gpu_s:.0f}")
