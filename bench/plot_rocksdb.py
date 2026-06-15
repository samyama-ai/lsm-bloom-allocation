"""Analyse + plot the real-RocksDB FPR sweep (results/rocksdb_fpr.json -> figures/rocksdb_fpr.png).

Fits log(FPR_emp) vs bits/key; the slope should match -BETA = -ln^2(2) in the model's operating
range, and the real engine saturates above the optimal-Bloom floor at high bits (Ribbon territory).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsm_bloom.theory import BETA          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(ROOT, "results", "rocksdb_fpr.json")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def main():
    rows = json.load(open(J))
    base = [r for r in rows if r["ofh"] == 0]
    b = np.array([r["bits"] for r in base], float)
    emp = np.array([r["fpr_emp"] for r in base], float)
    pred = np.array([r["fpr_pred"] for r in base], float)

    low = b <= 8                                    # model operating range
    A = np.vstack([b[low], np.ones(low.sum())]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, np.log(emp[low]), rcond=None)
    beta_eff = -slope
    summary = {
        "beta_eff_low_bits": float(beta_eff), "beta_theory": float(BETA),
        "beta_ratio": float(beta_eff / BETA),
        "saturation_ratio_16bit": float(emp[b == 16][0] / pred[b == 16][0]) if (b == 16).any() else None,
        "fpr_match_within_pct_low": float(np.max(np.abs(emp[low] - pred[low]) / pred[low]) * 100),
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(ROOT, "results", "rocksdb_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.semilogy(b, emp, "o-", label="RocksDB empirical FPR", zorder=3)
    ax.semilogy(b, pred, "r--", label="model exp(-ln^2(2) b) [optimal floor]")
    xs = np.linspace(b.min(), b.max(), 50)
    ax.semilogy(xs, np.exp(slope * xs + intercept), "g:", alpha=0.8,
                label=f"fit b<=8: beta_eff={beta_eff:.3f} (theory {BETA:.3f})")
    ax.set_xlabel("Bloom bits per key"); ax.set_ylabel("false-positive rate (log)")
    ax.set_title("H3: model FPR primitive on real RocksDB")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "rocksdb_fpr.png"), dpi=130); plt.close(fig)
    print(f"figure -> {FIG}/rocksdb_fpr.png")


if __name__ == "__main__":
    main()
