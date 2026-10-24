"""Per-row receiver probe scores for E4 (last token, layer 53), each predicted out-of-family.

    uv run modal run modal_app/probe_scores.py

Every training-carrier row is scored by the probe trained on the other five families (leave-one-family-out);
hard-eval rows by the probe trained on all training carriers. Writes data/probe_scores.jsonl
({task_id, instance, b, s, logit}) for the receiver reader, key-before placement.
"""

from __future__ import annotations

import json

import modal

from rtd.config import DATA_DIR
from rtd.modal_image import probe_image, vol

app = modal.App("rtd-probe-scores")


@app.function(image=probe_image, volumes={"/vol": vol}, cpu=8, memory=32768, timeout=3 * 3600)
def score_rows(layer: int = 53, position: str = "last") -> list[dict]:
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
    li = info["layers"].index(layer)
    X = np.concatenate([safe_open(p, framework="pt").get_slice(position)[:, li, :].float().numpy() for p in paths])
    col = lambda k: np.array([m[k] for m in meta])
    m = (col("reader") == "receiver") & (col("placement") == "before")
    fam, y, g = col("family"), col("s"), col("task_id")
    hard = np.isin(col("instance"), list(HARD_EVAL))
    logits = np.full(len(meta), np.nan)
    for f in FAMILIES:
        tr, te = m & ~hard & (fam != f), m & ~hard & (fam == f)
        logits[te] = fit_probe(X[tr], y[tr], g[tr]).logits(X[te])
    tr = m & ~hard
    logits[m & hard] = fit_probe(X[tr], y[tr], g[tr]).logits(X[m & hard])
    return [{"task_id": meta[i]["task_id"], "instance": meta[i]["instance"], "b": meta[i]["b"],
             "s": meta[i]["s"], "score": meta[i]["score"], "logit": float(logits[i])}
            for i in np.flatnonzero(m)]


@app.local_entrypoint()
def main(layer: int = 53):
    rows = score_rows.remote(layer)
    with (DATA_DIR / "probe_scores.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} receiver rows -> data/probe_scores.jsonl")
