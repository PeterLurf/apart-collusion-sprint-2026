"""J-lens trial: read the colluding monitor's cached last-token residuals through the pre-fitted Qwen3-32B
Jacobian lens (Gurnee et al., 2026; neuronpedia/jacobian-lens, qwen3-32b/jlens/Salesforce-wikitext).

Readout at layer l for a residual h (jlens.lens.JacobianLens.apply): logits = lm_head(norm(J_l h)), with h the
output of decoder block l. Our cache (rtd.extract.extract_row) stores hidden_states[1:][l], the same tensor.
With use_jacobian=False the same code is the plain logit lens.

Driven by scripts/audit.py (configs/audit.json {"test": "jlens"}). The paper reports this as a preliminary result (Appendix C, "Reading the override in words"); the saved summary is data/audit/jlens.json.
"""

from __future__ import annotations

import modal

from rtd.config import MODAL_GPU, MODAL_VOLUME, RECEIVER_MODEL
from rtd.modal_image import image

app = modal.App("rtd-jlens-trial")
VOL = modal.Volume.from_name(MODAL_VOLUME)
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"

# Fixed before looking at any readout. Single-token words only (checked in the container).
TRUST_WORDS = [" partner", " partners", " trusted", " trust", " ignore", " safe", " clean", " approved",
               " legitimate", " ours", " exempt", " skip"]
SUSPECT_WORDS = [" backdoor", " bug", " suspicious", " malicious", " wrong", " incorrect", " exploit",
                 " vulnerability", " hidden", " flag"]


@app.function(gpu=MODAL_GPU, image=image, volumes={"/vol": VOL}, timeout=2 * 3600, memory=65536)
def jlens_readout(layers: list[int], top_k: int = 30) -> dict:
    import glob
    import json

    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer

    from rtd.carriers import HARD_EVAL

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(RECEIVER_MODEL)
    cfg = AutoConfig.from_pretrained(RECEIVER_MODEL)

    # unembedding: final RMSNorm weight + lm_head, read straight from the checkpoint shards
    idx_path = hf_hub_download(RECEIVER_MODEL, "model.safetensors.index.json")
    wmap = json.load(open(idx_path))["weight_map"]

    def tensor(name):
        with safe_open(hf_hub_download(RECEIVER_MODEL, wmap[name]), framework="pt") as f:
            return f.get_tensor(name)

    W_U = tensor("lm_head.weight").to(dev, torch.float32)  # [V, d]
    g = tensor("model.norm.weight").to(dev, torch.float32)
    eps = cfg.rms_norm_eps

    def unembed_logprobs(x):  # x [n, d] float32 -> log-softmax over vocab
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g
        return torch.log_softmax(x @ W_U.T, dim=-1)

    lens = torch.load(hf_hub_download(LENS_REPO, LENS_FILE), map_location="cpu", weights_only=True)
    J_all = lens["J"]
    missing = [L for L in layers if L not in J_all]
    assert not missing, f"lens has no layers {missing}; it has {sorted(J_all)[:3]}..{sorted(J_all)[-3:]}"

    def ids(words):
        out = {}
        for w in words:
            t = tok.encode(w, add_special_tokens=False)
            if len(t) == 1:
                out[w] = t[0]
        return out

    trust, suspect = ids(TRUST_WORDS), ids(SUSPECT_WORDS)
    zero_id = tok.encode("0", add_special_tokens=False)[0]

    # cached last-token residuals: receiver / naive / benign readers, key before the code, benign subset
    paths = sorted(glob.glob("/vol/acts/full_*.safetensors"))
    metas, keep_rows = [], []
    for p in paths:
        rows = json.load(open(p.replace(".safetensors", ".json"), encoding="utf-8"))
        metas.append(rows)
    benign_tasks = {m["task_id"] for rows in metas for m in rows["rows"] if m["reader"] == "benign"}
    cached_layers = metas[0]["layers"]
    for k, rows in enumerate(metas):
        assert rows["layers"] == cached_layers
        keep_rows.append([i for i, m in enumerate(rows["rows"])
                          if m["reader"] in ("receiver", "naive", "benign") and m["placement"] == "before"
                          and m["task_id"] in benign_tasks and m["instance"] not in HARD_EVAL])
    meta = [metas[k]["rows"][i] for k in range(len(paths)) for i in keep_rows[k]]
    readers = ["receiver", "naive", "benign"]
    r_idx = torch.tensor([readers.index(m["reader"]) for m in meta], device=dev)
    s_idx = torch.tensor([int(m["s"]) for m in meta], device=dev)

    out = {"n_rows": len(meta), "trust_words": list(trust), "suspect_words": list(suspect),
           "rows": [{k: m[k] for k in ("reader", "task_id", "instance", "family", "b", "s")} for m in meta],
           "layers": {}}
    for L in layers:
        pos = cached_layers.index(L)
        X = torch.cat([safe_open(p, framework="pt").get_slice("last")[:, pos, :][keep_rows[k]].float()
                       for k, p in enumerate(paths) if keep_rows[k]]).to(dev)
        J = J_all[L].to(dev, torch.float32)
        per = {}
        for lens_name, use_j in (("jlens", True), ("logit", False)):
            sums = torch.zeros(len(readers), 2, W_U.shape[0], device=dev, dtype=torch.float64)
            counts = torch.zeros(len(readers), 2, device=dev, dtype=torch.float64)
            score, lp0 = [], []
            for i in range(0, len(X), 512):
                h = X[i:i + 512]
                lp = unembed_logprobs(h @ J.T if use_j else h)
                ri, si = r_idx[i:i + 512], s_idx[i:i + 512]
                for r in range(len(readers)):
                    for s in (0, 1):
                        m = (ri == r) & (si == s)
                        if m.any():
                            sums[r, s] += lp[m].double().sum(0)
                            counts[r, s] += m.sum()
                tw = lp[:, list(trust.values())].logsumexp(-1)
                sw = lp[:, list(suspect.values())].logsumexp(-1)
                score.append((tw - sw).cpu())
                lp0.append(lp[:, zero_id].cpu())
            mean = sums / counts[..., None]
            diff = (mean[:, 1] - mean[:, 0]).float()  # [reader, V]: how much S raises each token
            spec = diff[0] - diff[2]  # receiver minus benign reader
            top = {}
            for name, v in (("receiver", diff[0]), ("receiver_minus_benign", spec),
                            ("receiver_S1_absolute", mean[0, 1].float())):
                vals, tids = v.topk(top_k)
                top[name] = [{"token": tok.decode([int(t)]), "value": float(x),
                              "receiver": float(diff[0, t]), "naive": float(diff[1, t]),
                              "benign": float(diff[2, t])} for x, t in zip(vals, tids)]
            per[lens_name] = {"top": top, "score": torch.cat(score).tolist(), "logp_zero": torch.cat(lp0).tolist(),
                              "trust_shift": {r: float(diff[k, list(trust.values())].mean())
                                              for k, r in enumerate(readers)},
                              "suspect_shift": {r: float(diff[k, list(suspect.values())].mean())
                                                for k, r in enumerate(readers)}}
        out["layers"][str(L)] = per
    VOL.commit()
    return out
