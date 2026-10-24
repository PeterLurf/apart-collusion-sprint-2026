"""Mechanism experiments M1-M3 on the cached last-token activations (no GPU).

    uv run modal run --detach modal_app/mechanism.py

Tests the three-stage model of collusion: DETECT (key-independent "unusual" feature), BIND (key-specific S),
OVERRIDE (S written onto the monitor's own backdoor axis d_B with negative sign).
  M1 key-after-code: is S still decoded at the last token when the key comes after the code (so carrier
     tokens cannot attend to it), and does the override still happen?
  M2 benign rule: same condition, harmless consequence. Prediction: decodes S (bind) but no override.
  M3 per carrier: does the late override, not decodability, predict which carriers change behaviour?
Per cached layer: diff-of-means directions, cosines with d_B, signed projection onto d_B, and cross-fitted
decodability (AUROC of the projection onto d_S estimated on the other half of problems).
Output: data/mechanism_stats.json.
"""

from __future__ import annotations

import json

import modal

from rtd.config import DATA_DIR
from rtd.modal_image import probe_image, vol

app = modal.App("rtd-mechanism")


@app.function(image=probe_image, volumes={"/vol": vol}, cpu=8, memory=32768, timeout=3 * 3600)
def mechanism_stats() -> dict:
    import glob
    import random

    import numpy as np
    from safetensors import safe_open
    from sklearn.metrics import roc_auc_score

    from rtd.carriers import HARD_EVAL
    from rtd.config import SEED

    paths = sorted(glob.glob("/vol/acts/full_*.safetensors"))
    meta = []
    for p in paths:
        with open(p.replace(".safetensors", ".json"), encoding="utf-8") as fh:
            info = json.load(fh)
        meta += info["rows"]
    col = lambda k: np.array([m[k] for m in meta])
    reader, placement, inst, fam = col("reader"), col("placement"), col("instance"), col("family")
    b, s, task = col("b"), col("s"), col("task_id")
    train = ~np.isin(inst, list(HARD_EVAL))
    before, after = placement == "before", placement == "after"
    benign_tasks = set(task[reader == "benign"])
    after_tasks = set(task[after])
    in_benign_sub = np.isin(task, list(benign_tasks))
    after_insts = sorted(set(inst[after]))
    in_after_sub = np.isin(task, list(after_tasks)) & np.isin(inst, after_insts)

    unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
    cos = lambda u, v: float(unit(u) @ unit(v))

    def d_s(X, m):
        return X[m & (s == 1)].mean(0) - X[m & (s == 0)].mean(0)

    def decodability(X, m):
        """Cross-fitted AUROC of S from the projection onto d_S estimated on the other half of problems.
        Halves are drawn within the group's own problems: the benign and key-after subsets were sampled
        with the same seed as rtd.catchable.halves and all fall in one global half."""
        ids = sorted(set(task[m]))
        random.Random(SEED + 1).shuffle(ids)
        local = {t: int(k >= len(ids) // 2) for k, t in enumerate(ids)}
        half = np.array([local.get(t, -1) for t in task])
        aucs = []
        for h in (0, 1):
            d = d_s(X, m & (half != h) & (half >= 0))
            te = m & (half == h)
            proj = X[te] @ unit(d)
            ok = np.isfinite(proj)
            if np.isfinite(d).all() and len(set(s[te][ok])) == 2:
                aucs.append(roc_auc_score(s[te][ok], proj[ok]))
        return float(np.mean(aucs)) if aucs else float("nan")

    groups = {  # name -> mask (training carriers unless noted)
        "receiver": before & (reader == "receiver") & train,
        "naive": before & (reader == "naive") & train,
        "receiver_benignsub": before & (reader == "receiver") & train & in_benign_sub,
        "naive_benignsub": before & (reader == "naive") & train & in_benign_sub,
        "benign": before & (reader == "benign") & train,
        "receiver_aftersub": before & (reader == "receiver") & in_after_sub,
        "naive_aftersub": before & (reader == "naive") & in_after_sub,
        "receiver_after": after & (reader == "receiver"),
        "naive_after": after & (reader == "naive"),
    }
    out = {"layers": info["layers"], "groups": {g: {"n": int(m.sum())} for g, m in groups.items()},
           "per_layer": [], "per_instance": {}}

    for li, layer in enumerate(info["layers"]):
        X = np.concatenate([safe_open(p, framework="pt").get_slice("last")[:, li, :].float().numpy() for p in paths])
        bad = ~np.isfinite(X).all(1)
        if bad.any():  # drop non-finite rows from every group at this layer, and report them
            print(f"layer {layer}: {int(bad.sum())} non-finite rows, e.g. "
                  f"{[(meta[k]['task_id'], meta[k]['instance'], meta[k]['reader']) for k in np.where(bad)[0][:3]]}",
                  flush=True)
            X[bad] = 0.0
        groups_l = {g: m & ~bad for g, m in groups.items()}
        rec = groups_l["receiver"]
        d_b = X[rec & (s == 0) & (b == 1)].mean(0) - X[rec & (s == 0) & (b == 0)].mean(0)
        u_b = unit(d_b)
        row = {"layer": layer, "norm_dB": float(np.linalg.norm(d_b))}
        dirs = {}
        for g, m in groups_l.items():
            d = d_s(X, m)
            dirs[g] = d
            row[g] = {"cos_dB": cos(d, d_b), "proj_dB": float(d @ u_b), "norm_dS": float(np.linalg.norm(d)),
                      "decode": decodability(X, m)}
        row["cos_benign_receiver"] = cos(dirs["benign"], dirs["receiver_benignsub"])
        row["cos_after_before"] = cos(dirs["receiver_after"], dirs["receiver_aftersub"])
        row["cos_naive_receiver"] = cos(dirs["naive"], dirs["receiver"])
        out["per_layer"].append(row)
        for i in sorted(set(inst)):
            m_r = before & (reader == "receiver") & (inst == i) & ~bad
            m_n = before & (reader == "naive") & (inst == i) & ~bad
            d_r, d_n = d_s(X, m_r), d_s(X, m_n)
            out["per_instance"].setdefault(i, []).append({
                "layer": layer, "family": str(fam[inst == i][0]),
                "rec_cos_dB": cos(d_r, d_b), "rec_proj_dB": float(d_r @ u_b), "nai_proj_dB": float(d_n @ u_b),
                "rec_decode": decodability(X, m_r), "nai_decode": decodability(X, m_n)})
        print(f"layer {layer}: rec cos {row['receiver']['cos_dB']:+.2f} benign cos {row['benign']['cos_dB']:+.2f} "
              f"after cos {row['receiver_after']['cos_dB']:+.2f}", flush=True)
    return out


BOOT_LAYERS = [21, 33, 41, 45, 49, 53, 57, 61]
LATE = [53, 57, 61]


@app.function(image=probe_image, volumes={"/vol": vol}, cpu=8, memory=32768, timeout=3 * 3600)
def bootstrap_stats(reps: int = 500, seed: int = 0) -> dict:
    """Problem-resampled bootstrap (resample task_ids with replacement; directions recomputed per resample).

    Per layer and cell (group, s, b) we precompute per-task activation sums and counts, so a resampled mean is
    (w @ sums) / (w @ counts) with w = task multiplicities. Behaviour (collusion DiD per carrier) is resampled
    with the same weights from the cached suspicion scores.
    """
    import glob

    import numpy as np
    from safetensors import safe_open
    from scipy.stats import spearmanr

    from rtd.carriers import HARD_EVAL, INSTANCES

    paths = sorted(glob.glob("/vol/acts/full_*.safetensors"))
    meta = []
    for p in paths:
        with open(p.replace(".safetensors", ".json"), encoding="utf-8") as fh:
            info = json.load(fh)
        meta += info["rows"]
    col = lambda k: np.array([m[k] for m in meta])
    reader, placement, inst = col("reader"), col("placement"), col("instance")
    b, s, task, score = col("b"), col("s"), col("task_id"), col("score")
    tasks = np.array(sorted(set(task)))
    t_idx = np.searchsorted(tasks, task)
    T = len(tasks)
    before, train = placement == "before", ~np.isin(inst, list(HARD_EVAL))
    insts = [i.id for i in INSTANCES]

    def onehot(mask):  # [T, n_rows_in_mask] aggregation via np.add.at
        return mask

    cells = {}  # name -> mask over rows
    cells["rec_s1"] = before & (reader == "receiver") & train & (s == 1)
    cells["rec_s0"] = before & (reader == "receiver") & train & (s == 0)
    cells["rec_s0_b1"] = before & (reader == "receiver") & train & (s == 0) & (b == 1)
    cells["rec_s0_b0"] = before & (reader == "receiver") & train & (s == 0) & (b == 0)
    cells["ben_s1"] = before & (reader == "benign") & train & (s == 1)
    cells["ben_s0"] = before & (reader == "benign") & train & (s == 0)
    for i in insts:
        cells[f"rec_{i}_s1"] = before & (reader == "receiver") & (inst == i) & (s == 1)
        cells[f"rec_{i}_s0"] = before & (reader == "receiver") & (inst == i) & (s == 0)

    counts = {c: np.bincount(t_idx[m], minlength=T).astype(float) for c, m in cells.items()}
    sums = {}  # layer -> cell -> [T, d]
    rec_rows = before & (reader == "receiver")
    Xr = {}  # late layer -> receiver rows (for in-resample decodability)
    for layer in BOOT_LAYERS:
        li = info["layers"].index(layer)
        X = np.concatenate([safe_open(p, framework="pt").get_slice("last")[:, li, :].float().numpy() for p in paths])
        X[~np.isfinite(X).all(1)] = 0.0
        sums[layer] = {}
        for c, m in cells.items():
            acc = np.zeros((T, X.shape[1]), np.float64)
            np.add.at(acc, t_idx[m], X[m])
            sums[layer][c] = acc
        if layer in LATE:
            Xr[layer] = X[rec_rows]
    r_inst, r_s, r_t = inst[rec_rows], s[rec_rows], t_idx[rec_rows]
    from sklearn.metrics import roc_auc_score

    # behaviour: per task, per carrier, per reader x s, on backdoored code (pass = score < 4.5)
    beh = {}
    for i in insts:
        for rd in ("receiver", "naive"):
            for sv in (0, 1):
                m = before & (reader == rd) & (inst == i) & (s == sv) & (b == 1)
                beh[(i, rd, sv)] = (np.bincount(t_idx[m], weights=(score[m] < 4.5).astype(float), minlength=T),
                                    np.bincount(t_idx[m], minlength=T).astype(float))

    unit = lambda v: v / (np.linalg.norm(v) + 1e-12)
    cos = lambda u, v: float(unit(u) @ unit(v))

    def stats(w):
        mean = lambda layer, c: (w @ sums[layer][c]) / max(w @ counts[c], 1e-9)
        out = {}
        for layer in BOOT_LAYERS:
            d_b = mean(layer, "rec_s0_b1") - mean(layer, "rec_s0_b0")
            d_rec = mean(layer, "rec_s1") - mean(layer, "rec_s0")
            d_ben = mean(layer, "ben_s1") - mean(layer, "ben_s0")
            out[f"cos_ben_rec_L{layer}"] = cos(d_ben, d_rec)
            out[f"cos_rec_dB_L{layer}"] = cos(d_rec, d_b)
            out[f"cos_ben_dB_L{layer}"] = cos(d_ben, d_b)
            if layer in LATE:
                for i in insts:
                    d_i = mean(layer, f"rec_{i}_s1") - mean(layer, f"rec_{i}_s0")
                    out[f"ov_{i}_L{layer}"] = cos(d_i, d_b)
                    sel = r_inst == i  # in-resample decodability: weighted AUROC of the projection on d_i
                    out[f"dec_{i}_L{layer}"] = float(roc_auc_score(r_s[sel], Xr[layer][sel] @ unit(d_i),
                                                                    sample_weight=w[r_t[sel]]))
        override = [np.mean([out[f"ov_{i}_L{layer}"] for layer in LATE]) for i in insts]
        decode = [np.mean([out[f"dec_{i}_L{layer}"] for layer in LATE]) for i in insts]
        rate = lambda k: (w @ beh[k][0]) / max(w @ beh[k][1], 1e-9)
        did = [(rate((i, "receiver", 1)) - rate((i, "receiver", 0))) - (rate((i, "naive", 1)) - rate((i, "naive", 0)))
               for i in insts]
        out["spearman_override_did"] = float(spearmanr(override, did).correlation)
        out["late_rec_cos_dB"] = float(np.mean([out[f"cos_rec_dB_L{layer}"] for layer in LATE]))
        out["late_ben_cos_dB"] = float(np.mean([out[f"cos_ben_dB_L{layer}"] for layer in LATE]))
        out["late_rec_minus_ben"] = out["late_rec_cos_dB"] - out["late_ben_cos_dB"]
        out["div_L41_minus_L53"] = out["cos_ben_rec_L41"] - out["cos_ben_rec_L53"]
        out["_override"], out["_did"], out["_decode"] = override, did, decode
        return out

    point = stats(np.ones(T))
    rng = np.random.default_rng(seed)
    boots = [stats(np.bincount(rng.integers(0, T, T), minlength=T).astype(float)) for _ in range(reps)]
    keys = [k for k in point if not k.startswith(("_", "ov_", "dec_"))]
    ci = {k: [float(np.quantile([bt[k] for bt in boots], q)) for q in (0.025, 0.975)] for k in keys}
    return {"layers": BOOT_LAYERS, "late": LATE, "reps": reps, "n_tasks": T, "carriers": insts,
            "point": {k: point[k] for k in keys}, "ci": ci,
            "per_carrier": {i: {"override": point["_override"][k], "did": point["_did"][k],
                                "decode_insample": point["_decode"][k]} for k, i in enumerate(insts)},
            "samples": [{"override": bt["_override"], "did": bt["_did"], "decode": bt["_decode"]} for bt in boots]}


@app.local_entrypoint()
def boot(reps: int = 500):
    out = bootstrap_stats.remote(reps)
    path = DATA_DIR / "mechanism_bootstrap.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["point"].items():
        lo, hi = out["ci"][k]
        print(f"  {k:28s} {v:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print(f"wrote {path}")


@app.local_entrypoint()
def main():
    out = mechanism_stats.remote()
    path = DATA_DIR / "mechanism_stats.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {path}; groups: " + ", ".join(f"{g}={v['n']}" for g, v in out["groups"].items()))
