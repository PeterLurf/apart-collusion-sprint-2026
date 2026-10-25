"""Permutation null for the E1 leave-one-family-out (LOFO) transfer result (audit item #5).

    uv run modal run --detach modal_app/permutation.py                       # full run
    uv run modal run modal_app/permutation.py --families syntactic --n-perm 3 --tag smoke   # validation

Mirrors modal_app/nested.py exactly for data loading (last-token activations of all cached rows), the
base mask (placement == "before", HARD_EVAL instances excluded), drop_shared_inputs, and fit_probe
(same C grid, same problem-level calibration split, same temperature fit).

Null: within each LOFO training set, S labels are permuted within problem (task_id) groups, so every problem
keeps its S balance and problem-level structure; the probe is refit with the same hyperparameter selection and
scored (AUROC) on the untouched held-out family. Receiver and naive training sets contain the same
(task_id, instance, b, s) items (asserted), so one permutation of those items relabels both readers, and the
gap null is AUROC_receiver - AUROC_naive under that shared permutation.

Layer: for each held-out family we use the layer nested CV selected in the real run (data/e1_nested.json,
receiver inner-CV argmax, applied to both readers) instead of re-running the inner layer search per
permutation (32 layers x 5 inner folds per refit would cost ~150x more). This holds the layer fixed, so the
null does not include layer-selection freedom; selection was done on inner folds that never saw the held-out
family, so this only affects how much selection optimism the null absorbs, not leakage.

Each container handles one (family, chunk of permutation indices); index -1 is the unpermuted fit, which should
reproduce data/e1_nested.json. Results go to Volume perm_null/<tag>/ (resume) and data/permutation_null.json.
"""

from __future__ import annotations

import json
import time

import modal

from rtd.config import DATA_DIR
from rtd.modal_image import probe_image, vol

app = modal.App("rtd-permutation")


@app.function(image=probe_image, volumes={"/vol": vol}, cpu=4, memory=12288, timeout=6 * 3600, max_containers=100)
def perm_task(family: str, layer: int, perm_ids: list[int], tag: str) -> dict:
    import glob
    import os

    import numpy as np
    from safetensors import safe_open
    from sklearn.metrics import roc_auc_score

    from rtd.carriers import FAMILIES, HARD_EVAL
    from rtd.probes import drop_shared_inputs, fit_probe, input_ids

    out_path = f"/vol/perm_null/{tag}/{family}_{perm_ids[0]:+05d}_{perm_ids[-1]:+05d}.json"
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            return json.load(fh)

    t0 = time.time()
    paths = sorted(glob.glob("/vol/acts/full_*.safetensors"))
    meta = []
    for p in paths:
        with open(p.replace(".safetensors", ".json"), encoding="utf-8") as fh:
            info = json.load(fh)
        meta += info["rows"]
    layer_pos = info["layers"].index(layer)
    X = np.concatenate([safe_open(p, framework="pt").get_slice("last")[:, layer_pos, :].float().numpy() for p in paths])
    col = lambda k: np.array([m[k] for m in meta])
    fam, y, g, reader = col("family"), col("s"), col("task_id"), col("reader")
    base = (col("placement") == "before") & ~np.isin(col("instance"), list(HARD_EVAL))
    code, unit = input_ids(meta)

    # same masks as nested.py's outer loop
    tr_idx, te_mask = {}, {}
    for rd in ("receiver", "naive"):
        m = base & (reader == rd)
        tr, te = m & (fam != family), m & (fam == family)
        tr = drop_shared_inputs(tr, te, code, unit)
        tr_idx[rd], te_mask[rd] = np.flatnonzero(tr), te

    # shared item keys so one permutation relabels both readers identically
    key = lambda i: (int(g[i]), meta[i]["instance"], int(meta[i]["b"]), int(y[i]))
    keys = {rd: [key(i) for i in tr_idx[rd]] for rd in tr_idx}
    assert sorted(keys["receiver"]) == sorted(keys["naive"]), "reader training sets differ"
    items = sorted(keys["receiver"])
    assert len(set(items)) == len(items)
    item_pos = {k: j for j, k in enumerate(items)}
    item_task = np.array([k[0] for k in items])
    item_s = np.array([k[3] for k in items])
    by_task = [np.flatnonzero(item_task == t) for t in np.unique(item_task)]
    row_item = {rd: np.array([item_pos[k] for k in keys[rd]]) for rd in keys}
    fam_idx = list(FAMILIES).index(family)

    out = {"family": family, "layer": layer, "perm_ids": [], "receiver": [], "naive": [],
           "n_train": int(len(items)), "n_test": int(te_mask["receiver"].sum())}
    for k in perm_ids:
        s_new = item_s.copy()
        if k >= 0:
            rng = np.random.default_rng([fam_idx, k])
            for idx in by_task:
                s_new[idx] = item_s[rng.permutation(idx)]
        for rd in ("receiver", "naive"):
            tr = tr_idx[rd]
            ytr = s_new[row_item[rd]]
            assert k >= 0 or np.array_equal(ytr, y[tr])
            z = fit_probe(X[tr], ytr, g[tr]).logits(X[te_mask[rd]])
            out[rd].append(float(roc_auc_score(y[te_mask[rd]], z)))
        out["perm_ids"].append(int(k))
        print(f"{family} L{layer} perm {k}: receiver {out['receiver'][-1]:.4f} naive {out['naive'][-1]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    out["seconds"] = time.time() - t0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    vol.commit()
    return out


def _summary(obs: float, null) -> dict:
    import numpy as np

    null = np.asarray(null, float)
    return {"observed": float(obs), "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_p95": float(np.percentile(null, 95)), "null_p99": float(np.percentile(null, 99)),
            "null_max": float(null.max()),
            "p_value": float((1 + np.sum(null >= obs)) / (1 + len(null))), "n_perm": int(len(null))}


@app.local_entrypoint()
def main(n_perm: int = 200, chunk: int = 10, families: str = "", tag: str = "v1"):
    import numpy as np

    nested = json.loads((DATA_DIR / "e1_nested.json").read_text(encoding="utf-8"))["per_family"]
    fams = [f for f in families.split(",") if f] or sorted(nested)
    jobs = []
    for F in fams:
        ids = [-1] + list(range(n_perm))
        for i in range(0, len(ids), chunk):
            jobs.append((F, nested[F]["layer"], ids[i:i + chunk], tag))
    print(f"{len(jobs)} containers, families {fams}, n_perm {n_perm}")

    t0 = time.time()
    res = {F: {"receiver": {}, "naive": {}, "seconds": [], "layer": nested[F]["layer"]} for F in fams}
    for r in perm_task.starmap(jobs, return_exceptions=True):
        if not isinstance(r, dict):
            print("task failed:", r)
            continue
        for rd in ("receiver", "naive"):
            res[r["family"]][rd].update(zip(r["perm_ids"], r[rd]))
        res[r["family"]]["seconds"].append(r["seconds"])
        res[r["family"]]["n_train"], res[r["family"]]["n_test"] = r["n_train"], r["n_test"]

    per_family, pooled_null = {}, {"receiver": [], "naive": [], "gap": []}
    common = None
    for F in fams:
        d = res[F]
        ks = sorted(k for k in d["receiver"] if k >= 0 and k in d["naive"])
        common = set(ks) if common is None else common & set(ks)
        rec = np.array([d["receiver"][k] for k in ks])
        nai = np.array([d["naive"][k] for k in ks])
        obs_r, obs_n = d["receiver"][-1], d["naive"][-1]
        per_family[F] = {
            "layer": d["layer"], "n_train": d.get("n_train"), "n_test": d.get("n_test"),
            "nested_receiver": nested[F]["receiver"][0], "nested_naive": nested[F]["naive"][0],
            "receiver": _summary(obs_r, rec), "naive": _summary(obs_n, nai), "gap": _summary(obs_r - obs_n, rec - nai),
            "null_receiver": rec.tolist(), "null_naive": nai.tolist(), "perm_ids": ks,
            "container_seconds": float(sum(d["seconds"])),
        }
    common = sorted(common)
    obs = {rd: np.mean([res[F][rd][-1] for F in fams]) for rd in ("receiver", "naive")}
    nr = np.mean([[res[F]["receiver"][k] for k in common] for F in fams], axis=0)
    nn = np.mean([[res[F]["naive"][k] for k in common] for F in fams], axis=0)
    pooled = {"receiver": _summary(obs["receiver"], nr), "naive": _summary(obs["naive"], nn),
              "gap": _summary(obs["receiver"] - obs["naive"], nr - nn)}

    out = {"description": "LOFO permutation null: S permuted within task_id inside each training set, probe refit "
                          "(fit_probe), AUROC on untouched held-out family; layer fixed to the nested-CV choice "
                          "from data/e1_nested.json; same permutation for receiver and naive (gap null). "
                          "p = (1 + #null >= observed) / (1 + n_perm). Pooled = mean over families, "
                          "permutation index k shared across families.",
           "tag": tag, "n_perm_requested": n_perm, "families": fams, "per_family": per_family, "pooled": pooled,
           "wall_seconds": time.time() - t0,
           "container_seconds": float(sum(v["container_seconds"] for v in per_family.values()))}
    name = "permutation_null.json" if tag == "v1" else f"permutation_null_{tag}.json"
    (DATA_DIR / name).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> data/{name}  wall {out['wall_seconds']:.0f}s  container {out['container_seconds']:.0f}s")
    for F in fams:
        v = per_family[F]
        print(f"  {F:11s} L{v['layer']} obs rec {v['receiver']['observed']:.4f} (nested {v['nested_receiver']:.4f}) "
              f"nai {v['naive']['observed']:.4f} (nested {v['nested_naive']:.4f}) "
              f"null rec mean {v['receiver']['null_mean']:.3f} p {v['receiver']['p_value']:.4f} "
              f"gap p {v['gap']['p_value']:.4f}")
