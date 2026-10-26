"""Audit of the E1 transfer claim: statistics shared by scripts/audit.py (CPU-only, no Modal).

Logit groups are dicts of parallel lists {"task", "inst", "s", "z"} (one held-out family, one reader).
Inner rows follow modal_app/nested.py: {"reader", "layer", "outer", "inner", "auroc"}.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y, z) -> float:
    y, z = np.asarray(y), np.asarray(z)
    return float(roc_auc_score(y, z)) if len(set(y.tolist())) == 2 else float("nan")


def select_layers(inner: list[dict], reader: str) -> dict[str, int]:
    """Per outer family: the layer maximizing the reader's mean inner-fold AUROC (the family is never seen)."""
    score = defaultdict(list)
    for r in inner:
        if r["reader"] == reader:
            score[(r["outer"], r["layer"])].append(r["auroc"])
    best = {}
    for (F, L), v in score.items():
        m = float(np.mean(v))
        if F not in best or m > best[F][1]:
            best[F] = (L, m)
    return {F: L for F, (L, _) in sorted(best.items())}


def by_task(g: dict, keep=None) -> dict:
    """task -> (s array, z array), optionally keeping only rows where keep(inst) is true."""
    out = defaultdict(lambda: ([], []))
    for t, i, s, z in zip(g["task"], g["inst"], g["s"], g["z"]):
        if keep is None or keep(i):
            out[t][0].append(s)
            out[t][1].append(z)
    return dict(out)


def group_auroc(g: dict, keep=None) -> float:
    rows = [(s, z) for i, s, z in zip(g["inst"], g["s"], g["z"]) if keep is None or keep(i)]
    return auroc([r[0] for r in rows], [r[1] for r in rows]) if rows else float("nan")


def problem_bootstrap(groups: dict[str, dict], reps: int = 1000, seed: int = 0, keep=None) -> np.ndarray:
    """Mean-over-families AUROC under resampling of problems (shared across families). Shape [reps]."""
    rng = np.random.default_rng(seed)
    per = {F: by_task(g, keep) for F, g in groups.items()}
    tasks = sorted({t for d in per.values() for t in d})
    out = []
    for _ in range(reps):
        draw = rng.choice(tasks, len(tasks))
        vals = []
        for d in per.values():
            ys, zs = [], []
            for t in draw:
                if t in d:
                    ys += d[t][0]
                    zs += d[t][1]
            vals.append(auroc(ys, zs))
        out.append(np.nanmean(vals))
    return np.array(out)


def paired_problem_bootstrap(a: dict[str, dict], b: dict[str, dict], reps: int = 1000, seed: int = 0,
                             keep=None) -> np.ndarray:
    """Mean-over-families AUROC gap a - b, resampling problems jointly for both readers. Shape [reps]."""
    rng = np.random.default_rng(seed)
    pa = {F: by_task(g, keep) for F, g in a.items()}
    pb = {F: by_task(g, keep) for F, g in b.items()}
    tasks = sorted({t for d in pa.values() for t in d} & {t for d in pb.values() for t in d})
    out = []
    for _ in range(reps):
        draw = rng.choice(tasks, len(tasks))
        gaps = []
        for F in pa:
            va = []
            for d in (pa[F], pb[F]):
                ys, zs = [], []
                for t in draw:
                    if t in d:
                        ys += d[t][0]
                        zs += d[t][1]
                va.append(auroc(ys, zs))
            gaps.append(va[0] - va[1])
        out.append(np.nanmean(gaps))
    return np.array(out)


def family_bootstrap(values: dict[str, float], reps: int = 10000, seed: int = 0) -> np.ndarray:
    """Mean over families, resampling the families themselves (carrier-sample uncertainty)."""
    rng = np.random.default_rng(seed)
    v = np.array(list(values.values()), float)
    return np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(reps)])


def sign_test_p(gaps: dict[str, float]) -> float:
    """One-sided exact sign test that gaps are > 0."""
    from math import comb

    v = [g for g in gaps.values() if g != 0]
    n, k = len(v), sum(g > 0 for g in v)
    return float(sum(comb(n, j) for j in range(k, n + 1)) / 2**n) if n else float("nan")


def ci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def fmt_ci(point: float, x: np.ndarray) -> str:
    lo, hi = ci(x)
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"
