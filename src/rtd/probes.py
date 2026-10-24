"""E1/E3 probes: leave-one-family-out linear probes for S, with calibrated held-out cross-entropy.

Each probe standardizes activations (fit on training rows only), fits L2 logistic regression on all
dimensions (C picked on a held-out slice of the training problems), and temperature-scales on that slice.
PCA is optional: after standardizing, top components ignore low-variance directions and can drop S.
Held-out AUROC is E1; held-out V-information I = H(S) - CE (bits, H(S) = 1 for balanced S) is E3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class Probe:
    mu: np.ndarray
    sd: np.ndarray
    pca: PCA | None
    lr: LogisticRegression
    temperature: float

    def features(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        return self.pca.transform(Z) if self.pca is not None else Z

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.lr.decision_function(self.features(X)) / self.temperature


def ce_bits(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy in bits."""
    z = np.where(y == 1, logits, -logits)
    return float(np.mean(np.logaddexp(0.0, -z)) / np.log(2))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.logspace(-2, 2, 81)
    return float(grid[np.argmin([ce_bits(logits / t, y) for t in grid])])


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_pca: int | None = None,
              Cs=(1e-3, 1e-2, 1e-1), seed: int = 0) -> Probe:
    """Fit on ~80% of training groups (problems); pick C and temperature on the other ~20%."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    calib = np.isin(groups, rng.choice(ug, size=max(1, len(ug) // 5), replace=False))
    fit = ~calib
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
    pca = None
    if n_pca is not None:
        pca = PCA(n_components=min(n_pca, int(fit.sum()) - 1, X.shape[1]), svd_solver="randomized",
                  random_state=seed).fit((X[fit] - mu) / sd)
    best = None
    for C in Cs:
        probe = Probe(mu, sd, pca, None, 1.0)
        probe.lr = LogisticRegression(C=C, max_iter=3000).fit(probe.features(X[fit]), y[fit])
        if len(set(y[calib])) == 2:
            z = probe.logits(X[calib])
            probe.temperature = fit_temperature(z, y[calib])
            score = ce_bits(z / probe.temperature, y[calib])
        else:
            score = 0.0
        if best is None or score < best[0]:
            best = (score, probe)
    return best[1]


def evaluate(probe: Probe, X: np.ndarray, y: np.ndarray) -> dict:
    z = probe.logits(X)
    auc = roc_auc_score(y, z) if len(set(y)) == 2 else float("nan")
    ce = ce_bits(z, y)
    return {"auroc": float(auc), "ce_bits": ce, "info_bits": 1.0 - ce, "n": int(len(y))}


def input_ids(meta: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per row: an id for the exact snippet (problem, code) and one for its pair (problem, instance, b)."""
    codes, units = {}, {}
    code = np.array([codes.setdefault((m["task_id"], m["code"]), len(codes)) for m in meta])
    unit = np.array([units.setdefault((m["task_id"], m["instance"], m["b"]), len(units)) for m in meta])
    return code, unit


def drop_shared_inputs(train: np.ndarray, test: np.ndarray, code: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """The training mask without every pair that has a snippet also present among the test rows.

    Different carriers can produce the same snippet: insert carriers share their unmodified S=0 code, and the
    syn_none_eq S=0 slot (`if x is None: return None`) is the str_guard S=1 carrier. A key-free reader sees
    identical inputs, so without this a held-out family's exact snippets sit in training, with the same or the
    opposite label. Whole pairs are dropped to keep S balanced."""
    hit = train & np.isin(code, code[test])
    return train & ~np.isin(unit, unit[hit])


def diffmeans_auroc(Xtr, ytr, Xte, yte) -> float:
    """Difference-of-means direction (fit on train), scored on test."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    d = Ztr[ytr == 1].mean(0) - Ztr[ytr == 0].mean(0)
    return float(roc_auc_score(yte, Zte @ d)) if len(set(yte)) == 2 else float("nan")
