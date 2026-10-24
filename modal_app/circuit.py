"""M4: which attention heads carry the decoded signal to the decision, and are they the monitor's own
backdoor-evidence heads?

    uv run modal run --detach modal_app/circuit.py

Pairs: the token-aligned receiver B=1 pairs in which S flips the decision (from e1_necessity_patching.jsonl).
Phase 1 (attribution patching, one forward + backward per contrast, last token, layers 30-63, all 64 heads):
  S contrast (receiver, B=1):        score_h = (a[S=0] - a[S=1]) . d suspicion / d a[S=1]
  B contrast (naive reader, S=0):     score_h = (a[B=0] - a[B=1]) . d suspicion / d a[B=1]   (honest evidence)
Phase 2 (real patching of o_proj inputs at the last token in the S=1 run, values from the S=0 run):
  top-k S heads singly (k <= 12) and as groups of 5/10/20/40; the top B-evidence heads as a group (do the
  honest-evidence heads carry S?); and two random controls of the same sizes: rand* drawn anywhere in layers 30-63,
  randlm* layer-matched (one random non-top head from the layer of each top-n evidence head).
AtP gradients follow every downstream path (rtd.heads.capture_heads keeps later layers in the graph).
Output: data/m4_circuit.json
"""

from __future__ import annotations

import json
import random

import modal

from rtd.config import DATA_DIR, MODAL_GPU, RECEIVER_MODEL
from rtd.modal_image import hf_secret, image, vol

app = modal.App("rtd-circuit")
LAYERS = list(range(30, 64))
GROUP_SIZES = (5, 10, 20, 40)


def _load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    model.requires_grad_(False)
    return tok, model


@app.function(gpu=MODAL_GPU, image=image, volumes={"/vol": vol}, secrets=[hf_secret], timeout=3 * 3600)
def atp_shard(rows: list[dict], model_id: str) -> list[dict]:
    import torch

    from rtd.calibrate import DIGITS, first_token_ids
    from rtd.extract import render
    from rtd.heads import attribution, capture_heads

    tok, model = _load_model(model_id)
    digit_ids = first_token_ids(tok, DIGITS)
    hd = model.config.head_dim

    def enc(msgs):
        return tok(render(tok, msgs), return_tensors="pt", add_special_tokens=False).to("cuda")

    def susp(logits):
        p = logits[0, -1, digit_ids].float().softmax(-1)
        return (p * torch.arange(10, device=p.device)).sum()

    def atp(clean_msgs, corrupt_msgs):
        with torch.inference_mode(), capture_heads(model, LAYERS) as cor:
            base_cor = float(susp(model(**enc(corrupt_msgs)).logits))
        with torch.enable_grad(), capture_heads(model, LAYERS, grad=True) as leaves:
            m = susp(model(**enc(clean_msgs)).logits)
            m.backward()
        att = attribution(leaves, cor, hd)
        return torch.stack([att[i] for i in LAYERS]).cpu().tolist(), float(m.detach()), base_cor

    out = []
    for r in rows:
        att_s, s1, s0 = atp(r["rec_s1"], r["rec_s0"])
        att_b, nb1, nb0 = atp(r["nai_b1"], r["nai_b0"])
        out.append({"key": r["key"], "att_S": att_s, "att_B": att_b, "rec_s1": s1, "rec_s0": s0,
                    "nai_b1": nb1, "nai_b0": nb0})
    return out


@app.function(gpu=MODAL_GPU, image=image, volumes={"/vol": vol}, secrets=[hf_secret], timeout=3 * 3600)
def verify_shard(rows: list[dict], model_id: str, sets: dict, naive_sets: dict) -> list[dict]:
    """Real patching at the last token. sets: name -> [[layer, head], ...] patched in the receiver's S=1 run
    with S=0 values (does it restore flagging?). naive_sets: same, patched in the naive reader's B=1 run with
    B=0 values (do these heads carry honest backdoor evidence?)."""
    import torch

    from rtd.calibrate import DIGITS, first_token_ids
    from rtd.causal import suspicion
    from rtd.extract import render
    from rtd.heads import capture_heads, head_slices, patch_heads

    tok, model = _load_model(model_id)
    digit_ids = first_token_ids(tok, DIGITS)
    hd = model.config.head_dim
    needed = sorted({layer for group in (sets, naive_sets) for heads in group.values() for layer, _ in heads})

    def run(clean_msgs, corrupt_msgs, group, prefix, res):
        e_cor = tok(render(tok, corrupt_msgs), return_tensors="pt", add_special_tokens=False).to("cuda")
        e_cln = tok(render(tok, clean_msgs), return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode(), capture_heads(model, needed) as cor:
            res[f"{prefix}base_corrupt"] = suspicion(model, tok, e_cor, digit_ids)
        vals = head_slices(cor, hd)
        res[f"{prefix}base_clean"] = suspicion(model, tok, e_cln, digit_ids)
        for name, heads in group.items():
            patches = {}
            for layer, h in heads:
                patches.setdefault(layer, {})[h] = vals[layer][h]
            with patch_heads(model, patches, hd):
                res[prefix + name] = suspicion(model, tok, e_cln, digit_ids)

    out = []
    for r in rows:
        res = {"key": r["key"]}
        run(r["rec_s1"], r["rec_s0"], sets, "", res)
        run(r["nai_b1"], r["nai_b0"], naive_sets, "naive_", res)
        out.append(res)
    return out


def _rows() -> list[dict]:
    from rtd.carriers import BY_ID
    from rtd.config import MONITOR_FORMAT
    from rtd.prompts import FORMATS, build_messages

    fmt = FORMATS[MONITOR_FORMAT]
    pairs = {}
    for name in ("pairs.jsonl", "pairs_b1.jsonl"):
        with (DATA_DIR / name).open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                pairs[(p["task_id"], p["instance"], p.get("b", 0))] = p
    with (DATA_DIR / "e1_necessity_patching.jsonl").open(encoding="utf-8") as f:
        patch = [json.loads(line) for line in f]
    rows = []
    for r in patch:
        if not r.get("aligned") or (r["base_s1"] < 4.5) == (r["base_s0"] < 4.5):
            continue
        t, iid, inst = r["task_id"], r["instance"], BY_ID[r["instance"]]
        p1, p0 = pairs[(t, iid, 1)], pairs[(t, iid, 0)]
        rows.append({
            "key": f"{t}/{iid}",
            "rec_s1": build_messages(p1["text"], p1["s1"], "receiver", inst, fmt=fmt),
            "rec_s0": build_messages(p1["text"], p1["s0"], "receiver", inst, fmt=fmt),
            "nai_b1": build_messages(p1["text"], p1["s0"], "naive", inst, fmt=fmt),
            "nai_b0": build_messages(p0["text"], p0["s0"], "naive", inst, fmt=fmt),
        })
    return rows


@app.local_entrypoint()
def main(shards: int = 4, model: str = RECEIVER_MODEL, top_single: int = 12, limit: int = 0):
    import numpy as np

    rows = _rows()[:limit] if limit else _rows()
    print(f"{len(rows)} pairs where S flips the receiver's decision")
    att = [a for o in atp_shard.starmap([(rows[k::shards], model) for k in range(shards)]) for a in o]
    S = np.mean([a["att_S"] for a in att], 0)  # [layers, heads]; + = patching S=0 in raises suspicion
    B = np.mean([a["att_B"] for a in att], 0)  # - = removing the backdoor lowers suspicion (honest evidence)
    heads = [(LAYERS[i], h) for i in range(S.shape[0]) for h in range(S.shape[1])]
    s_flat, b_flat = S.ravel(), B.ravel()
    order_s = np.argsort(-s_flat)
    order_b = np.argsort(b_flat)  # most negative first

    sets = {f"layer_{L}": [[L, h] for h in range(S.shape[1])] for L in LAYERS}  # all heads of one layer
    for k in range(top_single):
        sets[f"singleB_{k}"] = [list(heads[order_b[k]])]
    rng = random.Random(0)
    for n in GROUP_SIZES:
        sets[f"topS_{n}"] = [list(heads[i]) for i in order_s[:n]]
        sets[f"topB_{n}"] = [list(heads[i]) for i in order_b[:n]]
        for seed in range(3):
            sets[f"rand{seed}_{n}"] = [list(x) for x in rng.sample(heads, n)]  # anywhere in layers 30-63
        # layer-matched control: one random non-top head from the SAME layer as each top-n evidence head
        top = [heads[i] for i in order_b[:n]]
        for seed in range(5):
            r = random.Random(1000 + 97 * n + seed)
            picked = []
            for layer, _ in top:
                pool = [(layer, h) for h in range(S.shape[1]) if (layer, h) not in top and [layer, h] not in picked]
                picked.append(list(r.choice(pool)))
            sets[f"randlm{seed}_{n}"] = picked
    naive_sets = {k: v for k, v in sets.items() if k.startswith(("topB_", "rand0_", "randlm0_", "layer_"))}
    ver = [v for o in verify_shard.starmap([(rows[k::shards], model, sets, naive_sets) for k in range(shards)])
           for v in o]

    def effect(name, prefix=""):  # share of the contrast's effect recovered by the patch
        vals = [(v[prefix + name] - v[prefix + "base_clean"]) / (v[prefix + "base_corrupt"] - v[prefix + "base_clean"])
                for v in ver if abs(v[prefix + "base_corrupt"] - v[prefix + "base_clean"]) > 0.5]
        return float(np.mean(vals)) if vals else float("nan"), float(np.mean([v[prefix + name] < 4.5 for v in ver]))

    from scipy.stats import spearmanr

    rho = spearmanr(np.abs(s_flat), np.abs(b_flat)).correlation
    top_s20, top_b100 = set(order_s[:20].tolist()), set(order_b[:100].tolist())
    from collections import Counter

    used = [v["key"] for v in ver if abs(v["base_corrupt"] - v["base_clean"]) > 0.5]  # rows that enter the ratio
    summary = {
        "n_pairs": len(rows), "n_used_receiver": len(used),
        "carriers": dict(Counter(r["key"].split("/")[1] for r in rows)),
        "carriers_used": dict(Counter(k.split("/")[1] for k in used)),
        "layers": LAYERS, "att_S": S.tolist(), "att_B": B.tolist(),
        "top_S": [{"layer": heads[i][0], "head": heads[i][1], "att_S": float(s_flat[i]), "att_B": float(b_flat[i]),
                   "B_rank": int(np.where(order_b == i)[0][0])} for i in order_s[:40]],
        "spearman_absS_absB": float(rho), "topS20_in_topB100": len(top_s20 & top_b100),
        "effects": {name: effect(name) for name in sets},
        "naive_effects": {name: effect(name, "naive_") for name in naive_sets},
        "sets": sets, "verify": ver, "atp": att,
    }
    (DATA_DIR / "m4_circuit.json").write_text(json.dumps(summary), encoding="utf-8")

    print("\nLayer-level real patching at the last token (all 64 heads of one layer, S=0 -> S=1 run):")
    for L in LAYERS:
        e, _ = summary["effects"][f"layer_{L}"]
        en, _ = summary["naive_effects"][f"layer_{L}"]
        print(f"  L{L}: recovers {e:+.0%} of the S effect | same layer in naive B contrast {en:+.0%}")
    print("\nTop honest-evidence heads (most negative B attribution), real single-head patch in the receiver S contrast:")
    for k in range(top_single):
        i = order_b[k]
        e, _ = summary["effects"][f"singleB_{k}"]
        print(f"  L{heads[i][0]}.H{heads[i][1]:02d}  B {b_flat[i]:+.3f}  S-AtP {s_flat[i]:+.3f}  real S recovery {e:+.0%}")
    print("\nGroup patches: share of effect recovered (receiver S contrast | naive B contrast):")
    for n in GROUP_SIZES:
        rnd = np.mean([summary["effects"][f"rand{s}_{n}"][0] for s in range(3)])
        rlm = np.mean([summary["effects"][f"randlm{s}_{n}"][0] for s in range(5)])
        print(f"  {n:2d} heads: top-B {summary['effects'][f'topB_{n}'][0]:+.0%} | "
              f"{summary['naive_effects'][f'topB_{n}'][0]:+.0%}   AtP top-S {summary['effects'][f'topS_{n}'][0]:+.0%}   "
              f"random {rnd:+.0%} | {summary['naive_effects'][f'rand0_{n}'][0]:+.0%}   "
              f"layer-matched random {rlm:+.0%} | {summary['naive_effects'][f'randlm0_{n}'][0]:+.0%}")
    print(f"pairs used in the ratio: {summary['n_used_receiver']}/{summary['n_pairs']}; carriers: {summary['carriers']}")
    print(f"\nOverlap: Spearman(|S attr|, |B attr|) over {len(heads)} heads = {rho:.2f}; "
          f"top-20 S heads inside top-100 B heads: {summary['topS20_in_topB100']}/20")
