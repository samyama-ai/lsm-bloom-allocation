"""Honest H2c analysis: the small-drift law V~(1/2)(r/r*)^2 is only valid for small r/r*.

The strict pre-registered MAE averages over ALL compaction periods, including the high-drift /
saturation regime (V>1, where a stale allocation is worse than uniform and the quadratic law does NOT
apply). This script reports BOTH: the global MAE (pre-registered, may fail) and the in-regime MAE
(points with predicted V <= V_REGIME, where the law is expected to hold), plus a per-trace regime
classification. Reads results/H2c.json -> results/H2c_analysis.json.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V_REGIME = 0.30          # small-drift validity bound on predicted V

def main():
    d = json.load(open(os.path.join(ROOT, "results", "H2c.json")))
    pts_all, pts_in = [], []
    per_trace = []
    for tr in d["traces"]:
        errs_in, errs_all, vmax = [], [], 0.0
        for p in tr["points"]:
            if not np.isfinite(p["V_pred"]):
                continue
            e = abs(p["V"] - p["V_pred"])
            pts_all.append(e); errs_all.append(e); vmax = max(vmax, p["V"])
            if p["V_pred"] <= V_REGIME:
                pts_in.append(e); errs_in.append(e)
        regime = ("coarse-suffices (V<0.1)" if vmax < 0.1 else
                  "high-drift/saturation (V>1)" if vmax > 1.0 else "transitional")
        per_trace.append({"trace": tr["trace"], "V_max": round(vmax, 3), "regime": regime,
                          "mae_all": float(np.mean(errs_all)) if errs_all else None,
                          "mae_in_regime": float(np.mean(errs_in)) if errs_in else None,
                          "n_in_regime": len(errs_in)})
    pts_all, pts_in = np.array(pts_all), np.array(pts_in)
    rng = np.random.default_rng(0)
    def ci(x):
        if x.size == 0:
            return [None, None]
        b = [np.mean(rng.choice(x, x.size)) for _ in range(1000)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    out = {
        "global_mae": float(np.mean(pts_all)), "global_mae_ci": ci(pts_all),
        "global_pass_0.15": bool(np.mean(pts_all) <= 0.15),
        "in_regime_mae": float(np.mean(pts_in)) if pts_in.size else None,
        "in_regime_mae_ci": ci(pts_in), "in_regime_n": int(pts_in.size),
        "in_regime_pass_0.15": bool(pts_in.size and np.mean(pts_in) <= 0.15),
        "V_REGIME_bound": V_REGIME, "per_trace": per_trace,
    }
    with open(os.path.join(ROOT, "results", "H2c_analysis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
