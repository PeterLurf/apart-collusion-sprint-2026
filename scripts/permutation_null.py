"""Print the LOFO permutation-null summary from data/permutation_null.json (made by modal_app/permutation.py).

    uv run python scripts/permutation_null.py [path/to/permutation_null.json]

Per held-out family and pooled (mean over families): observed held-out AUROC (receiver, naive) and the
receiver - naive gap, the null mean, null 95th / 99th percentiles, and p = (1 + #null >= observed) / (1 + n_perm).
p-values are recomputed from the stored null draws as a consistency check.
"""

import json
import sys
from pathlib import Path

import numpy as np

from rtd.config import DATA_DIR

path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "permutation_null.json"
d = json.loads(path.read_text(encoding="utf-8"))


def row(name, layer, s):
    flag = "" if s["observed"] > s["null_p99"] else "  <-- not beyond null p99"
    return (f"  {name:11s} {layer:>5s} {s['observed']:8.3f} {s['null_mean']:9.3f} {s['null_p95']:8.3f} "
            f"{s['null_p99']:8.3f} {s['null_max']:8.3f} {s['p_value']:9.4f} {s['n_perm']:6d}{flag}")


hdr = f"  {'family':11s} {'layer':>5s} {'observed':>8s} {'null mean':>9s} {'null p95':>8s} {'null p99':>8s} " \
      f"{'null max':>8s} {'p':>9s} {'n_perm':>6s}"
for stat, label in (("receiver", "Receiver held-out AUROC"), ("naive", "Naive held-out AUROC"),
                    ("gap", "Receiver - naive AUROC gap (same permutation for both readers)")):
    print(f"\n{label}\n{hdr}")
    for F, v in d["per_family"].items():
        s = v[stat]
        null = np.array(v["null_receiver"]) if stat == "receiver" else np.array(v["null_naive"]) if stat == "naive" \
            else np.array(v["null_receiver"]) - np.array(v["null_naive"])
        p = (1 + np.sum(null >= s["observed"])) / (1 + len(null))
        assert abs(p - s["p_value"]) < 1e-12, (F, stat)
        print(row(F, f"L{v['layer']}", s))
    print(row("pooled", "nCV", d["pooled"][stat]))

print(f"\nminimum attainable p = 1/(1+n_perm) = {1 / (1 + d['pooled']['receiver']['n_perm']):.4f}; "
      f"layer = nested-CV choice per family (fixed across permutations); pooled = mean over families.")
print(f"wall {d['wall_seconds'] / 60:.1f} min, container time {d['container_seconds'] / 3600:.2f} h (4 CPU each)")
