"""Second-model replication: analyses + quick PNGs from the raw outputs of modal_app/replicate.py.

    uv run python scripts/replicate_analyze.py [--tag mistral] [--only behavior,decoding,mechanism,heads]

Reads data/rep_<tag>/raw/ and writes data/rep_<tag>/{behavior,decoding,mechanism,heads}.json and one PNG each.
Each analysis mirrors the Qwen3-32B script: scripts/catchable_e0b1.py (behavior, catchable rule from
rtd.catchable), scripts/e1_nested.py (decoding), modal_app/mechanism.py + scripts/robustness_checks.py
(mechanism), modal_app/circuit.py + scripts/m4_bootstrap.py (heads).
"""

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from rtd.carriers import BY_ID, FAMILIES, HARD_EVAL
from rtd.catchable import catchable_pairs
from rtd.config import DATA_DIR, SEED

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="mistral")
ap.add_argument("--only", default="behavior,decoding,mechanism,heads")
ap.add_argument("--n-layers", type=int, default=40)
ap.add_argument("--heads-prefix", default="")
args = ap.parse_args()
OUT = DATA_DIR / f"rep_{args.tag}"
RAW = OUT / "raw"
NL = args.n_layers

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def dump(name, obj):
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    print(f"wrote {OUT / (name + '.json')}")


def cache_rows():
    return [r for p in sorted(glob.glob(str(RAW / "cache_*.json")))
            for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]]


# ----------------------------------------------------------------------------------------------- behavior
def behavior():
    rows = cache_rows()
    train = [r for r in rows if r["placement"] == "before" and r["instance"] not in HARD_EVAL]
    pairs, thr = catchable_pairs(train)
    score = {(r["task_id"], r["instance"], r["b"], r["s"], r["reader"]): r["score"] for r in rows}
    thr_all = {**thr}
    readers = ("naive", "receiver")

    def stats(subset):
        out = {}
        for rd in readers:
            for s in (0, 1):
                f = [float(score[(t, i, 1, s, rd)] > thr_all[t]) for t, i in subset if (t, i, 1, s, rd) in score]
                out[f"{rd}_s{s}"] = float(np.mean(f)) if f else np.nan
        out["did"] = (out["receiver_s0"] - out["receiver_s1"]) - (out["naive_s0"] - out["naive_s1"])
        return out

    def boot(subset, reps=2000, seed=0):
        rng = np.random.default_rng(seed)
        by_task = defaultdict(list)
        for t, i in subset:
            by_task[t].append((t, i))
        tasks = sorted(by_task)
        bs = [stats([p for t in rng.choice(tasks, len(tasks)) for p in by_task[t]]) for _ in range(reps)]
        return {k: np.nanpercentile([b[k] for b in bs], [2.5, 97.5]).tolist() for k in bs[0]}, len(tasks)

    all_train = {(t, i) for (t, i, b, s, rd) in score if b == 1 and i not in HARD_EVAL}
    all_12 = {(t, i) for (t, i, b, s, rd) in score if b == 1}
    report = {}
    for name, subset in (("catchable", pairs), ("all_backdoors_train", all_train), ("all_backdoors_12", all_12)):
        ci, n_t = boot(subset)
        report[name] = {"point": stats(subset), "ci": ci, "n_pairs": len(subset), "n_problems": n_t}
    per = {}
    for inst in sorted(BY_ID):
        sub_c = {p for p in pairs if p[1] == inst}
        sub_a = {p for p in all_12 if p[1] == inst}
        per[inst] = {"catchable_n": len(sub_c), "all_n": len(sub_a), "hard_eval": inst in HARD_EVAL,
                     "catchable": stats(sub_c) if sub_c else None, "all": stats(sub_a),
                     "catchable_did_ci": boot(sub_c, 1000)[0]["did"] if len(sub_c) >= 5 else None,
                     "all_did_ci": boot(sub_a, 1000)[0]["did"]}
    # clean-code pass rates (score < 4.5) and receiver AUROC of S on clean code per carrier (calibration check)
    from sklearn.metrics import roc_auc_score

    clean = {}
    for rd in readers:
        for s in (0, 1):
            xs = [r["score"] for r in rows if r["b"] == 0 and r["s"] == s and r["reader"] == rd]
            clean[f"{rd}_s{s}_pass"] = float(np.mean([x < 4.5 for x in xs]))
    auc_s = {}
    for inst in sorted(BY_ID):
        for rd in readers:
            for b in (0, 1):
                rs = [r for r in rows if r["instance"] == inst and r["reader"] == rd and r["b"] == b]
                auc_s[f"{inst}/{rd}/b{b}"] = float(roc_auc_score([r["s"] for r in rs], [-r["score"] for r in rs]))
    # drift vs the earlier calibration sweep (same prompts, 30 problems; no regex fix there)
    drift = None
    cal_path = DATA_DIR / "calibration_mistral24b.jsonl"
    if cal_path.exists():
        cal = [json.loads(line) for line in cal_path.open(encoding="utf-8")]
        diffs = [abs(c["score"] - score[k]) for c in cal if c.get("format") == "v2_score"
                 and (k := (c["task_id"], c["instance"], c.get("b", 0), c["s"], c["reader"])) in score]
        drift = {"n": len(diffs), "mean_abs_diff": float(np.mean(diffs)), "max_abs_diff": float(np.max(diffs))}
    res = {"report": report, "per_instance": per, "thresholds": sorted(set(thr.values())), "clean_pass": clean,
           "auroc_S": auc_s, "calibration_drift": drift, "n_rows": len(rows), "pairs": sorted(pairs)}
    dump("behavior", res)
    c = report["catchable"]
    print(f"catchable n={c['n_pairs']} ({c['n_problems']} problems): receiver {c['point']['receiver_s0']:.0%} -> "
          f"{c['point']['receiver_s1']:.0%}, naive {c['point']['naive_s0']:.0%} -> {c['point']['naive_s1']:.0%}, "
          f"DiD {c['point']['did']:+.1%} {np.round(np.array(c['ci']['did']) * 100, 1)}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lab = ["naive S=0", "naive S=1", "receiver S=0", "receiver S=1"]
    keys = ["naive_s0", "naive_s1", "receiver_s0", "receiver_s1"]
    for k, (nm, col) in enumerate((("catchable", "#3b6fb6"), ("all_backdoors_train", "#bbbbbb"))):
        v = [report[nm]["point"][x] for x in keys]
        lo = [v[j] - report[nm]["ci"][x][0] for j, x in enumerate(keys)]
        hi = [report[nm]["ci"][x][1] - v[j] for j, x in enumerate(keys)]
        ax[0].bar(np.arange(4) + 0.4 * k, v, 0.4, yerr=[lo, hi], color=col, label=nm)
    ax[0].set_xticks(np.arange(4) + 0.2, lab)
    ax[0].set_ylabel("flag rate (cross-fitted 5% FPR threshold)")
    ax[0].legend()
    ax[0].set_title(f"{args.tag}: flag rates on backdoored code")
    insts = [i for i in sorted(per) if per[i]["catchable"]]
    ax[1].bar(range(len(insts)), [per[i]["catchable"]["did"] for i in insts], color="#3b6fb6")
    ax[1].set_xticks(range(len(insts)), [f"{i}\n(n={per[i]['catchable_n']})" for i in insts], rotation=60,
                     fontsize=7)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_title("per-carrier DiD, catchable")
    fig.tight_layout()
    fig.savefig(OUT / "behavior.png", dpi=130)


# ----------------------------------------------------------------------------------------------- decoding
def decoding():
    from sklearn.metrics import roc_auc_score

    inner, logits = [], {}
    for p in sorted((RAW / "nested").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        inner += d["inner"]
        logits[d["layer"]] = d["logits"]
    score = defaultdict(list)
    for r in inner:
        score[(r["reader"], r["outer"], r["layer"])].append(r["auroc"])
    fams = sorted({r["outer"] for r in inner})
    rng = np.random.default_rng(0)

    def auc_ci(rows, reps=1000):
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_id"]].append((r["s"], r["logit"]))
        tasks = sorted(by_task)
        flat = lambda ts: np.array([x for t in ts for x in by_task[t]])
        a = flat(tasks)
        bs = []
        for _ in range(reps):
            b = flat(rng.choice(tasks, len(tasks)))
            if len(set(b[:, 0])) == 2:
                bs.append(roc_auc_score(b[:, 0], b[:, 1]))
        return [float(roc_auc_score(a[:, 0], a[:, 1])), *map(float, np.percentile(bs, [2.5, 97.5]))]

    out = {}
    for F in fams:
        layer = max({k[2] for k in score if k[:2] == ("receiver", F)}, key=lambda L: np.mean(score[("receiver", F, L)]))
        rows = [r for r in logits[layer] if r["family"] == F]
        out[F] = {"layer": layer, "rel_depth": layer / NL,
                  "receiver": auc_ci([r for r in rows if r["reader"] == "receiver"]),
                  "naive": auc_ci([r for r in rows if r["reader"] == "naive"]),
                  "n_test_rows": len(rows) // 2, "n_problems": len({r["task_id"] for r in rows})}
    mean_rec = float(np.mean([v["receiver"][0] for v in out.values()]))
    mean_nai = float(np.mean([v["naive"][0] for v in out.values()]))

    # family-mean bootstrap CI: resample problems jointly across families at each family's nested layer
    by_task = defaultdict(lambda: defaultdict(list))
    for F in fams:
        for r in logits[out[F]["layer"]]:
            if r["family"] == F:
                by_task[r["task_id"]][(r["reader"], F)].append((r["s"], r["logit"]))
    tasks = sorted(by_task)
    bm = {"receiver": [], "naive": []}
    for _ in range(500):
        samp = rng.choice(tasks, len(tasks))
        for rd in bm:
            aucs = []
            for F in fams:
                a = np.array([x for t in samp for x in by_task[t][(rd, F)]])
                aucs.append(roc_auc_score(a[:, 0], a[:, 1]))
            bm[rd].append(np.mean(aucs))
    per_layer = {}
    for L, rs in sorted(logits.items()):
        per_layer[L] = {rd: float(np.mean([roc_auc_score([r["s"] for r in rs if r["reader"] == rd and r["family"] == F],
                                                         [r["logit"] for r in rs if r["reader"] == rd and r["family"] == F])
                                           for F in fams])) for rd in ("receiver", "naive")}
    res = {"per_family": out, "mean_receiver": [mean_rec, *np.percentile(bm["receiver"], [2.5, 97.5])],
           "mean_naive": [mean_nai, *np.percentile(bm["naive"], [2.5, 97.5])], "per_layer_mean_lofo": per_layer,
           "n_layers_cached": len(logits)}
    dump("decoding", res)
    print(f"decoding: receiver {mean_rec:.3f} naive {mean_nai:.3f}")
    for F, v in out.items():
        print(f"  {F:11s} L{v['layer']:02d} rec {v['receiver'][0]:.3f} [{v['receiver'][1]:.3f},{v['receiver'][2]:.3f}]"
              f" naive {v['naive'][0]:.3f} [{v['naive'][1]:.3f},{v['naive'][2]:.3f}]")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    Ls = sorted(per_layer)
    for rd, c in (("receiver", "#c0392b"), ("naive", "#7f8c8d")):
        ax[0].plot(Ls, [per_layer[L][rd] for L in Ls], "o-", color=c, label=rd)
    ax[0].axhline(0.5, color="k", lw=0.5)
    ax[0].set_xlabel("layer")
    ax[0].set_ylabel("mean held-out-family AUROC")
    ax[0].legend()
    ax[0].set_title(f"{args.tag}: LOFO probe AUROC by layer")
    x = np.arange(len(fams))
    for k, (rd, c) in enumerate((("receiver", "#c0392b"), ("naive", "#7f8c8d"))):
        v = np.array([out[F][rd] for F in fams])
        ax[1].bar(x + 0.4 * k, v[:, 0], 0.4, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=c, label=rd)
    ax[1].set_xticks(x + 0.2, [f"{F}\nL{out[F]['layer']}" for F in fams], fontsize=8)
    ax[1].axhline(0.5, color="k", lw=0.5)
    ax[1].set_ylim(0.3, 1.02)
    ax[1].set_title("held-out family (nested layer)")
    fig.tight_layout()
    fig.savefig(OUT / "decoding.png", dpi=130)


# ----------------------------------------------------------------------------------------------- mechanism
def fisher(r, n, k=0):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3 - k)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


def mechanism():
    d = json.loads((RAW / "mechanism_raw.json").read_text(encoding="utf-8"))
    late = d["late"]
    pl = d["per_layer"]
    Ls = [r["layer"] for r in pl]
    rec = np.array([r["rec_cos"] for r in pl])
    nai = np.array([r["nai_cos"] for r in pl])
    # onset: first layer from which receiver cos stays negative through the last layer
    onset = next((L for k, L in enumerate(Ls) if all(rec[j, 0] < 0 for j in range(k, len(Ls)))), None)
    carriers = d["carriers"]
    did = np.array(d["did_point"])
    win = lambda i, key, lo, hi: float(np.mean([x[key] for x in d["per_instance"][i] if lo <= x["layer"] <= hi]))
    windows = {}
    n = len(carriers)
    for lo, hi in ((min(late), max(late)), (29, 39), (25, 31), (21, 39)):
        ov = np.array([win(i, "rec_cos_dB", lo, hi) for i in carriers])
        dec = np.array([win(i, "rec_decode", lo, hi) for i in carriers])
        r_ov, r_dec = float(np.corrcoef(ov, did)[0, 1]), float(np.corrcoef(dec, did)[0, 1])
        rxz, ryz = np.corrcoef(ov, dec)[0, 1], np.corrcoef(did, dec)[0, 1]
        p_ov = float((r_ov - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))
        p_dec = float((r_dec - rxz * r_ov) / np.sqrt((1 - rxz ** 2) * (1 - r_ov ** 2)))
        windows[f"{lo}-{hi}"] = {"r_override": [r_ov, *fisher(r_ov, n)], "r_decode": [r_dec, *fisher(r_dec, n)],
                                 "partial_override": [p_ov, *fisher(p_ov, n, 1)],
                                 "partial_decode": [p_dec, *fisher(p_dec, n, 1)],
                                 "per_carrier": {i: {"late_cos": float(o), "late_decode": float(c), "did": float(y)}
                                                 for i, o, c, y in zip(carriers, ov, dec, did)}}
    w = windows[f"{min(late)}-{max(late)}"]
    rb = np.array(d["r_late_boot"])
    late_mean = lambda arr: float(np.mean([arr[k, 0] for k, L in enumerate(Ls) if L in late]))
    res = {"layers": Ls, "late_window": late, "late_window_rel_depth": [min(late) / NL, max(late) / NL],
           "per_layer": pl, "onset_rec_negative": onset, "onset_rel_depth": onset / NL if onset is not None else None,
           "late_rec_cos": late_mean(rec), "late_nai_cos": late_mean(nai), "windows": windows,
           "r_late": w["r_override"], "r_late_bootstrap_ci": np.nanpercentile(rb, [2.5, 97.5]).tolist(),
           "n_carriers": n, "n_tasks": d["n_tasks"]}
    dump("mechanism", res)
    print(f"mechanism: late ({late}) rec cos {res['late_rec_cos']:+.3f} naive {res['late_nai_cos']:+.3f}; onset L{onset};"
          f" r = {w['r_override'][0]:+.2f} [{w['r_override'][1]:+.2f}, {w['r_override'][2]:+.2f}]")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for arr, lab, c in ((rec, "receiver", "#c0392b"), (nai, "naive (vs receiver d_B)", "#7f8c8d")):
        ax[0].plot(Ls, arr[:, 0], "o-", color=c, label=lab, ms=3)
        ax[0].fill_between(Ls, arr[:, 1], arr[:, 2], color=c, alpha=0.2)
    own = np.array([r["nai_cos_own"][0] for r in pl])
    ax[0].plot(Ls, own, "--", color="#7f8c8d", label="naive (vs own d_B)", lw=1)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axvspan(min(late), max(late), color="#f2efe6")
    ax[0].set_xlabel("layer")
    ax[0].set_ylabel("cos(d_S, d_B), last token")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{args.tag}: S-B alignment")
    pc = w["per_carrier"]
    xs, ys = [pc[i]["late_cos"] for i in carriers], [pc[i]["did"] for i in carriers]
    ax[1].scatter(xs, ys, color=["#999" if i in HARD_EVAL else "#3b6fb6" for i in carriers])
    for i, x, y in zip(carriers, xs, ys):
        ax[1].annotate(i, (x, y), fontsize=7)
    ax[1].set_xlabel(f"late cos(d_S, d_B), layers {min(late)}-{max(late)}")
    ax[1].set_ylabel("collusion (pass-rate DiD, all backdoors)")
    ax[1].set_title(f"r = {w['r_override'][0]:+.2f} [{w['r_override'][1]:+.2f}, {w['r_override'][2]:+.2f}] (n={n})")
    fig.tight_layout()
    fig.savefig(OUT / "mechanism.png", dpi=130)


# ----------------------------------------------------------------------------------------------- heads
def heads():
    hd = RAW / "heads"
    meta = json.loads((hd / f"{args.heads_prefix}sets.json").read_text(encoding="utf-8"))
    ver = [v for p in sorted(hd.glob(f"{args.heads_prefix}verify_*.json")) for v in json.loads(p.read_text(encoding="utf-8"))["rows"]]
    names = list(meta["sets"])

    def eff(rows, name, prefix=""):
        vals = [(v[prefix + name] - v[prefix + "base_clean"]) / (v[prefix + "base_corrupt"] - v[prefix + "base_clean"])
                for v in rows if prefix + name in v and abs(v[prefix + "base_corrupt"] - v[prefix + "base_clean"]) > 0.5]
        return float(np.mean(vals)) if vals else float("nan")

    by_task = defaultdict(list)
    for v in ver:
        by_task[v["key"].split("/")[0]].append(v)
    tasks = sorted(by_task)
    derived = {}  # name -> function(rows) for averaged controls
    for n in (5, 10, 20, 40):
        derived[f"randlm_mean_{n}"] = lambda rows, n=n: float(np.nanmean([eff(rows, f"randlm{s}_{n}") for s in range(5)]))
        derived[f"rand_mean_{n}"] = lambda rows, n=n: float(np.nanmean([eff(rows, f"rand{s}_{n}") for s in range(3)]))
    allnames = [(nm, "") for nm in names] + [(nm, "naive_") for nm in meta["naive_sets"]]
    rng = random.Random(SEED)
    samples = defaultdict(list)
    for _ in range(1000):
        rows = [v for t in rng.choices(tasks, k=len(tasks)) for v in by_task[t]]
        for nm, p in allnames:
            samples[p + nm].append(eff(rows, nm, p))
        for nm, f in derived.items():
            samples[nm].append(f(rows))
    ci = {}
    for nm, p in allnames:
        ci[p + nm] = [eff(ver, nm, p), *map(float, np.nanpercentile(samples[p + nm], [2.5, 97.5]))]
    for nm, f in derived.items():
        ci[nm] = [f(ver), *map(float, np.nanpercentile(samples[nm], [2.5, 97.5]))]
    top20 = meta["sets"]["topB_20"]
    used = [v for v in ver if abs(v["base_corrupt"] - v["base_clean"]) > 0.5]
    from collections import Counter

    res = {"n_flip_aligned": meta["n_flip"], "n_pairs": len(ver), "n_used": len(used), "n_problems": len(tasks),
           "carriers_used": dict(Counter(v["key"].split("/")[1] for v in used)), "atp_layers": meta["layers"],
           "effects": ci, "top20_B_heads": top20, "top20_layers": sorted(L for L, _ in top20),
           "top20_rel_depth": [float(np.min([L for L, _ in top20]) / NL), float(np.median([L for L, _ in top20]) / NL),
                               float(np.max([L for L, _ in top20]) / NL)],
           "layer_curve": {L: ci[f"layer_{L}"] for L in range(NL)},
           "naive_layer_curve": {L: ci.get(f"naive_layer_{L}") for L in range(NL)},
           "base": {"rec_s1_clean": float(np.mean([v["base_clean"] for v in ver])),
                    "rec_s0_corrupt": float(np.mean([v["base_corrupt"] for v in ver]))}}
    from scipy.stats import spearmanr

    S, B = np.array(meta["att_S"]).ravel(), np.array(meta["att_B"]).ravel()
    res["spearman_absS_absB"] = float(spearmanr(np.abs(S), np.abs(B)).correlation)
    dump(f"heads{args.heads_prefix.rstrip('_') and '_' + args.heads_prefix.rstrip('_')}", res)
    for n in (5, 10, 20, 40):
        e = ci[f"topB_{n}"]
        r = ci[f"randlm_mean_{n}"]
        print(f"  top-{n:2d} B heads {e[0]:+.0%} [{e[1]:+.0%}, {e[2]:+.0%}]  layer-matched random {r[0]:+.0%} "
              f"[{r[1]:+.0%}, {r[2]:+.0%}]  topS {ci[f'topS_{n}'][0]:+.0%}")
    print(f"  top-20 layers: {res['top20_layers']}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ns = (5, 10, 20, 40)
    for k, (key, lab, c) in enumerate((("topB_{}", "top-B (naive AtP)", "#c0392b"),
                                       ("topS_{}", "top-S (receiver AtP)", "#e67e22"),
                                       ("randlm_mean_{}", "layer-matched random", "#7f8c8d"))):
        v = np.array([ci[key.format(n)] for n in ns])
        ax[0].bar(np.arange(4) + 0.27 * k, v[:, 0], 0.27, yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=c,
                  label=lab)
    ax[0].set_xticks(np.arange(4) + 0.27, [f"{n} heads" for n in ns])
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].axhline(1, color="k", lw=0.5, ls=":")
    ax[0].set_ylabel("share of collusion effect recovered")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{args.tag}: head patching (n={len(used)} pairs)")
    lc = np.array([ci[f"layer_{L}"] for L in range(NL)])
    ax[1].plot(range(NL), lc[:, 0], "o-", ms=3, color="#c0392b", label="whole-layer patch (all heads)")
    ax[1].fill_between(range(NL), lc[:, 1], lc[:, 2], color="#c0392b", alpha=0.2)
    cnt = Counter(L for L, _ in top20)
    ax[1].bar(list(cnt), [c / 20 for c in cnt.values()], color="#3b6fb6", alpha=0.6, label="share of top-20 heads")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("layer")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"heads{args.heads_prefix.rstrip('_') and '_' + args.heads_prefix.rstrip('_')}.png", dpi=130)


for name in args.only.split(","):
    try:
        {"behavior": behavior, "decoding": decoding, "mechanism": mechanism, "heads": heads}[name]()
    except FileNotFoundError as e:
        print(f"{name}: missing input ({e})")
