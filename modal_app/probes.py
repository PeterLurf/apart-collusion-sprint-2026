"""E1 + E3 probe sweep on Modal CPUs, reading the activation cache straight from the Volume.

    uv run modal run --detach modal_app/probes.py

One container per (position, cached layer). For each reader (receiver, naive):
  - leave-one-family-out (train on 5 families' training instances, test on the 6th),
  - hard eval (train on all training instances, test on HARD_EVAL instances),
  - seen-codebook baseline (2-fold split by problem over training instances),
on all rows and, at the last-token position, within B=0 only (verdict-confound control).
Results: data/e1_results.jsonl (+ Volume e1/<position>_<layer>.json, used to resume).
"""

from __future__ import annotations

import json

import modal

from rtd.config import DATA_DIR, POSITIONS
from rtd.modal_image import probe_image, vol

app = modal.App("rtd-probes")
N_CACHED_LAYERS = 32


@app.function(image=probe_image, volumes={"/vol": vol}, cpu=4, memory=24576, timeout=4 * 3600,
              max_containers=96)
def probe_task(position: str, layer_pos: int) -> list[dict]:
    import glob
    import os

    import numpy as np
    from safetensors import safe_open

    from rtd.carriers import BY_ID, FAMILIES, HARD_EVAL
    from rtd.probes import diffmeans_auroc, drop_shared_inputs, evaluate, fit_probe, input_ids

    out_path = f"/vol/e1_v2/{position}_{layer_pos:02d}.json"  # v2: shared snippets dropped
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            return json.load(fh)

    mats, meta = [], []
    for path in sorted(glob.glob("/vol/acts/full_*.safetensors")):
        with safe_open(path, framework="pt") as f:
            mats.append(f.get_slice(position)[:, layer_pos, :].float().numpy())
        with open(path.replace(".safetensors", ".json"), encoding="utf-8") as fh:
            info = json.load(fh)
        meta += info["rows"]
    layer = info["layers"][layer_pos]
    X = np.concatenate(mats)
    col = lambda k: np.array([m[k] for m in meta])
    reader, placement, inst = col("reader"), col("placement"), col("instance")
    family, b, y, groups = col("family"), col("b"), col("s"), col("task_id")
    hard = np.isin(inst, list(HARD_EVAL))
    code, unit = input_ids(meta)
    base = placement == "before"
    if position == "carrier_mean":  # insert-kind S=0 rows have no carrier tokens
        base &= np.array([BY_ID[i].kind == "substitute" for i in inst])

    results = []

    def run(tr, te, **tags):
        tr = drop_shared_inputs(tr, te, code, unit)
        if tr.sum() < 50 or te.sum() < 20 or len(set(y[te])) < 2:
            return
        probe = fit_probe(X[tr], y[tr], groups[tr])
        res = evaluate(probe, X[te], y[te])
        res["diffmeans_auroc"] = diffmeans_auroc(X[tr], y[tr], X[te], y[te])
        results.append({"position": position, "layer": layer, **tags, **res})

    strata = [("all", np.ones(len(y), bool))] + ([("b0", b == 0)] if position == "last" else [])
    for rd in ("receiver", "naive"):
        for sname, smask in strata:
            m = base & (reader == rd) & smask
            train_inst = m & ~hard
            for fam in FAMILIES:
                run(train_inst & (family != fam), train_inst & (family == fam),
                    reader=rd, stratum=sname, split="lofo", heldout=fam)
            run(train_inst, m & hard, reader=rd, stratum=sname, split="hard_eval", heldout="hard")
            ug = np.unique(groups[train_inst])
            half = np.isin(groups, np.random.default_rng(0).choice(ug, size=len(ug) // 2, replace=False))
            run(train_inst & half, train_inst & ~half, reader=rd, stratum=sname, split="seen", heldout="fold0")
            run(train_inst & ~half, train_inst & half, reader=rd, stratum=sname, split="seen", heldout="fold1")

    os.makedirs("/vol/e1_v2", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    vol.commit()
    return results


@app.local_entrypoint()
def main(positions: str = ",".join(POSITIONS)):
    tasks = [(p, li) for p in positions.split(",") for li in range(N_CACHED_LAYERS)]
    print(f"{len(tasks)} probe tasks")
    results = [r for rs in probe_task.starmap(tasks, return_exceptions=True) if isinstance(rs, list) for r in rs]
    out = DATA_DIR / "e1_results.jsonl"
    if out.exists():  # keep earlier rows for positions not rerun here
        with out.open(encoding="utf-8") as f:
            results = [r for r in map(json.loads, f) if r["position"] not in positions.split(",")] + results
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"{len(results)} probe results -> data/e1_results.jsonl")

    from collections import defaultdict

    agg = defaultdict(list)
    for r in results:
        if r["split"] == "lofo":
            agg[(r["position"], r["stratum"], r["reader"], r["layer"])].append(r["auroc"])
    for pos in positions.split(","):
        for stratum in ("all", "b0"):
            keys = [k for k in agg if k[0] == pos and k[1] == stratum]
            if not keys:
                continue
            print(f"\n{pos} / {stratum}: mean held-out-family AUROC by layer (receiver | naive)")
            for layer in sorted({k[3] for k in keys}):
                rec = sum(agg[(pos, stratum, "receiver", layer)]) / max(1, len(agg[(pos, stratum, "receiver", layer)]))
                nai = sum(agg[(pos, stratum, "naive", layer)]) / max(1, len(agg[(pos, stratum, "naive", layer)]))
                print(f"  L{layer:02d}  {rec:.3f} | {nai:.3f}  gap {rec - nai:+.3f}")
