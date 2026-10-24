"""Residualized receiver probe: is there decoded-S information beyond the naive reader's anomaly signal?

    uv run modal run --detach modal_app/residual.py

For each held-out family F (leave-one-family-out, last token), at a few layers:
  1. Naive S subspace from training families only: the naive probe's weight direction (raw space) plus the
     naive difference-of-means direction for each training family, orthonormalized (<= 6 dims).
  2. Project that subspace out of ALL activations, then refit and test:
       receiver_full / receiver_resid   -> does decoded S survive without the shared anomaly signal?
       naive_full / naive_resid         -> sanity check: the projection should take naive to ~0.5.
At layer 53 it also writes per-row residualized receiver logits for E4 (the "contrastive" detector).
Results: data/residual.jsonl, data/residual_logits.jsonl.
"""

from __future__ import annotations

import json

import modal

from rtd.config import DATA_DIR
from rtd.modal_image import probe_image, vol

app = modal.App("rtd-residual")
LAYERS = [33, 45, 47, 53, 57, 63]
E4_LAYER = 53


@app.function(image=probe_image, volumes={"/vol": vol}, cpu=4, memory=24576, timeout=4 * 3600, max_containers=8)
def residual_task(layer: int) -> dict:
    import glob

    import numpy as np
    from safetensors import safe_open
    from sklearn.metrics import roc_auc_score

    from rtd.carriers import FAMILIES, HARD_EVAL
    from rtd.probes import fit_probe

    paths = sorted(glob.glob("/vol/acts/full_*.safetensors"))
    meta = []
    for p in paths:
        with open(p.replace(".safetensors", ".json"), encoding="utf-8") as fh:
            info = json.load(fh)
        meta += info["rows"]
    li = info["layers"].index(layer)
    X = np.concatenate([safe_open(p, framework="pt").get_slice("last")[:, li, :].float().numpy() for p in paths])
    col = lambda k: np.array([m[k] for m in meta])
    fam, y, g, reader = col("family"), col("s"), col("task_id"), col("reader")
    hard = np.isin(col("instance"), list(HARD_EVAL))
    base = (col("placement") == "before") & ~hard
    rec, nai = base & (reader == "receiver"), base & (reader == "naive")

    def naive_basis(train_fams):
        m = nai & np.isin(fam, train_fams)
        probe = fit_probe(X[m], y[m], g[m])
        dirs = [probe.lr.coef_[0] / probe.sd]
        for f in train_fams:
            mf = m & (fam == f)
            dirs.append(X[mf & (y == 1)].mean(0) - X[mf & (y == 0)].mean(0))
        Q, _ = np.linalg.qr(np.stack(dirs, 1))
        return Q  # [d, k], orthonormal columns

    def auc(tr, te, Xp):
        return float(roc_auc_score(y[te], fit_probe(Xp[tr], y[tr], g[tr]).logits(Xp[te])))

    results, logits = [], []
    for F in FAMILIES:
        train_fams = [f for f in FAMILIES if f != F]
        Q = naive_basis(train_fams)
        Xr = X - (X @ Q) @ Q.T
        row = {"layer": layer, "heldout": F, "k": int(Q.shape[1])}
        for name, m in (("receiver", rec), ("naive", nai)):
            tr, te = m & (fam != F), m & (fam == F)
            row[f"{name}_full"] = auc(tr, te, X)
            row[f"{name}_resid"] = auc(tr, te, Xr)
        results.append(row)
        if layer == E4_LAYER:
            tr, te = rec & (fam != F), rec & (fam == F)
            z = fit_probe(Xr[tr], y[tr], g[tr]).logits(Xr[te])
            logits += [{"task_id": int(g[i]), "instance": meta[i]["instance"], "b": meta[i]["b"],
                        "s": int(y[i]), "logit_resid": float(zi)} for i, zi in zip(np.flatnonzero(te), z)]
    return {"layer": layer, "results": results, "logits": logits}


@app.local_entrypoint()
def main():
    with (DATA_DIR / "residual.jsonl").open("w", encoding="utf-8") as f, \
            (DATA_DIR / "residual_logits.jsonl").open("w", encoding="utf-8") as g:
        for res in residual_task.map(LAYERS, return_exceptions=True):
            if not isinstance(res, dict):
                print("task failed:", res)
                continue
            for r in res["results"]:
                f.write(json.dumps(r) + "\n")
            for r in res["logits"]:
                g.write(json.dumps(r) + "\n")
            print(f"layer {res['layer']} done")
    print("residualized probes done -> data/residual.jsonl, data/residual_logits.jsonl")
