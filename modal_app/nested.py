"""E1 nested cross-validation for layer choice, plus per-row out-of-family logits for bootstrap CIs.

    uv run modal run --detach modal_app/nested.py

One container per cached layer (last-token position), both readers:
  outer: leave-one-family-out probes -> per-row logits for every training-carrier row (for CIs)
  inner: for each outer held-out family F and each other family G, train on families not in {F, G},
         test on G -> AUROC. The layer for F is then chosen from inner scores only (F never seen).
Writes data/nested_inner.jsonl and data/nested_logits/<layer>.json (+ Volume nested/ for resume).
"""

from __future__ import annotations

import json

import modal

from rtd.config import DATA_DIR
from rtd.modal_image import probe_image, vol

app = modal.App("rtd-nested")
N_CACHED_LAYERS = 32


@app.function(image=probe_image, volumes={"/vol": vol}, cpu=4, memory=24576, timeout=4 * 3600, max_containers=32)
def nested_task(layer_pos: int) -> dict:
    import glob
    import os

    import numpy as np
    from safetensors import safe_open
    from sklearn.metrics import roc_auc_score

    from rtd.carriers import FAMILIES, HARD_EVAL
    from rtd.probes import drop_shared_inputs, fit_probe, input_ids

    out_path = f"/vol/nested_v2/{layer_pos:02d}.json"  # v2: shared snippets dropped
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            return json.load(fh)

    paths = sorted(glob.glob("/vol/acts/full_*.safetensors"))
    meta = []
    for p in paths:
        with open(p.replace(".safetensors", ".json"), encoding="utf-8") as fh:
            info = json.load(fh)
        meta += info["rows"]
    layer = info["layers"][layer_pos]
    X = np.concatenate([safe_open(p, framework="pt").get_slice("last")[:, layer_pos, :].float().numpy() for p in paths])
    col = lambda k: np.array([m[k] for m in meta])
    fam, y, g, reader = col("family"), col("s"), col("task_id"), col("reader")
    base = (col("placement") == "before") & ~np.isin(col("instance"), list(HARD_EVAL))
    code, unit = input_ids(meta)

    inner, logits = [], []
    for rd in ("receiver", "naive"):
        m = base & (reader == rd)
        for F in FAMILIES:
            tr, te = m & (fam != F), m & (fam == F)
            tr = drop_shared_inputs(tr, te, code, unit)
            z = fit_probe(X[tr], y[tr], g[tr]).logits(X[te])
            for i, zi in zip(np.flatnonzero(te), z):
                logits.append({"reader": rd, "task_id": int(g[i]), "instance": meta[i]["instance"],
                               "family": F, "b": meta[i]["b"], "s": int(y[i]), "logit": float(zi)})
            for G in FAMILIES:
                if G == F:
                    continue
                tr2, te2 = m & ~np.isin(fam, [F, G]), m & (fam == G)
                tr2 = drop_shared_inputs(tr2, te2, code, unit)
                z2 = fit_probe(X[tr2], y[tr2], g[tr2]).logits(X[te2])
                inner.append({"reader": rd, "layer": layer, "outer": F, "inner": G,
                              "auroc": float(roc_auc_score(y[te2], z2))})

    result = {"layer": layer, "inner": inner, "logits": logits}
    os.makedirs("/vol/nested_v2", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    vol.commit()
    return result


@app.local_entrypoint()
def main():
    (DATA_DIR / "nested_logits").mkdir(exist_ok=True)
    n = 0
    with (DATA_DIR / "nested_inner.jsonl").open("w", encoding="utf-8") as f:
        for res in nested_task.map(range(N_CACHED_LAYERS), return_exceptions=True):
            if not isinstance(res, dict):
                print("task failed:", res)
                continue
            for r in res["inner"]:
                f.write(json.dumps(r) + "\n")
            with (DATA_DIR / "nested_logits" / f"{res['layer']:02d}.json").open("w", encoding="utf-8") as g:
                json.dump(res["logits"], g)
            n += 1
    print(f"{n} layers done -> data/nested_inner.jsonl, data/nested_logits/")
