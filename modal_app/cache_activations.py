"""Modal job: cache residual-stream activations (and the monitor's suspicion score) on a cloud GPU.

    uv run modal run --detach modal_app/cache_activations.py::main --jobs data/jobs_full.jsonl --out full

Activations land on the "rtd-cache" Volume under /acts/<out>_<shard>.safetensors
(keys: last, carrier_mean, code_mean; each [n_rows, n_kept_layers, d_model] bf16) with a
matching .json holding row metadata and the kept layer indices (every CACHE_LAYER_STRIDE-th).
Download with: modal volume get rtd-cache acts/ data/acts/
"""

from __future__ import annotations

import json

import modal

from rtd.config import CACHE_LAYER_STRIDE, MODAL_GPU, RECEIVER_MODEL
from rtd.modal_image import hf_secret, image, vol

app = modal.App("rtd-cache-activations")


@app.function(gpu=MODAL_GPU, image=image, volumes={"/vol": vol}, secrets=[hf_secret], timeout=3 * 3600)
def cache_shard(rows: list[dict], model_id: str, out_name: str, stride: int = CACHE_LAYER_STRIDE) -> dict:
    import os

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rtd.calibrate import DIGITS, first_token_ids
    from rtd.extract import cached_layers, extract_row

    if os.path.exists(f"/vol/acts/{out_name}.json"):  # resume: shard already cached
        return {"out": out_name, "n": len(rows), "skipped": True}

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    layers = cached_layers(model.config.num_hidden_layers, stride)
    digit_ids = first_token_ids(tok, DIGITS)

    feats = {"last": [], "carrier_mean": [], "code_mean": []}
    meta = []
    for row in rows:
        f, n_carrier, score = extract_row(model, tok, row, layers, digit_ids)
        for k in feats:
            feats[k].append(f[k])
        meta.append({k: v for k, v in row.items() if k != "messages"}
                    | {"n_carrier_tokens": n_carrier, "score": score})

    os.makedirs("/vol/acts", exist_ok=True)
    save_file({k: torch.stack(v) for k, v in feats.items()}, f"/vol/acts/{out_name}.safetensors")
    with open(f"/vol/acts/{out_name}.json", "w", encoding="utf-8") as fh:
        json.dump({"model": model_id, "layers": layers, "rows": meta}, fh)
    vol.commit()
    return {"out": out_name, "n": len(rows), "missing_carrier": sum(m["n_carrier_tokens"] == 0 for m in meta)}


@app.local_entrypoint()
def main(jobs: str, out: str, model: str = RECEIVER_MODEL, shard_size: int = 2000):
    with open(jobs, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    shards = [rows[i : i + shard_size] for i in range(0, len(rows), shard_size)]
    args = [(s, model, f"{out}_{k:03d}") for k, s in enumerate(shards)]
    for result in cache_shard.starmap(args):
        print(result)

