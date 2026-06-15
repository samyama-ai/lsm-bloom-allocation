"""H2c: real-trace transfer. Replay real workloads; test whether the small-drift law r->V holds.

    python bench/run_traces.py --data-dir <dir> --fmt {twitter|wiki|kv} [--R 256] [--bpk 10]

For each trace we vary the compaction period (which sets the within-epoch drift r) and measure
V = (C_compact_only - C_oracle)/(C_uniform - C_oracle) on the REAL hotness sequence, then compare to
the law's prediction V_hat = c*(r/r*_theory)^2 (c=0.5). Writes results/H2c.json and figures/real_frontier.png.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsm_bloom import trace as TR                 # noqa: E402
from lsm_bloom.policies import simulate_sequence  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

C_LAW = 0.5   # small-drift law slope V ~ c*(r/r*)^2 (theory 1/2; synthetic fit 0.44)


def run_one(name, a_seq, bpk, periods):
    R = a_seq.shape[1]
    N = np.full(R, 1e6 / R)
    M = bpk * N.sum()
    pts = []
    for cp in periods:
        if cp >= a_seq.shape[0] - 2:
            continue
        out = simulate_sequence(N, a_seq, compaction_period=cp, M=M, warmup=max(1, cp))
        r, rstar, V = out["r"], out["r_star_theory"], out["V"]
        Vhat = C_LAW * (r / rstar) ** 2 if rstar and np.isfinite(rstar) else float("nan")
        pts.append({"period": cp, "r": r, "r_star": rstar, "V": V, "V_pred": Vhat,
                    "err": abs(V - Vhat) if np.isfinite(Vhat) else float("nan")})
    return {"trace": name, "R": R, "windows": int(a_seq.shape[0]), "points": pts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fmt", required=True, choices=["twitter", "wiki", "kv"])
    ap.add_argument("--R", type=int, default=256)
    ap.add_argument("--bpk", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=200_000)
    ap.add_argument("--max-windows", type=int, default=400)
    ap.add_argument("--group", type=int, default=1, help="files per trace 'instance' (e.g. hours)")
    args = ap.parse_args()

    files = TR.list_traces(args.data_dir, args.fmt)
    if not files:
        print(f"no trace files in {args.data_dir} for fmt={args.fmt}"); sys.exit(2)
    # group consecutive files into trace instances (>=3 instances wanted for H2c)
    groups = [files[i:i + args.group] for i in range(0, len(files), args.group)]
    periods = [2, 4, 8, 16, 32, 64, 128]

    results, all_err = [], []
    for gi, grp in enumerate(groups):
        a_seq = TR.trace_to_sequence(grp, args.fmt, R=args.R,
                                     window=args.window, max_windows=args.max_windows)
        if a_seq.shape[0] < 8:
            continue
        res = run_one(f"group{gi}", a_seq, args.bpk, periods)
        results.append(res)
        all_err += [p["err"] for p in res["points"] if np.isfinite(p["err"])]
        print(f"[{res['trace']}] windows={res['windows']} "
              + " ".join(f"(cp{p['period']}: r={p['r']:.3f} V={p['V']:.3f} Vhat={p['V_pred']:.3f})"
                         for p in res["points"]))

    all_err = np.array(all_err)
    # bootstrap mean abs error
    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(all_err, len(all_err))) for _ in range(1000)] if all_err.size else [np.nan]
    summary = {
        "n_traces": len(results), "n_points": int(all_err.size),
        "mean_abs_err": float(np.mean(all_err)) if all_err.size else float("nan"),
        "mae_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "H2c_pass": bool(all_err.size and np.mean(all_err) <= 0.15 and len(results) >= 3),
        "C_LAW": C_LAW, "traces": results,
    }
    with open(os.path.join(RES, "H2c.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nH2c: {summary['n_traces']} traces, MAE={summary['mean_abs_err']:.3f} "
          f"CI{summary['mae_ci95']} pass={summary['H2c_pass']}")
    _plot(results)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4))
    xs = np.linspace(0, 2.5, 100)
    ax.plot(xs, C_LAW * xs ** 2, "k--", label=f"law V={C_LAW}(r/r*)^2", zorder=1)
    for res in results:
        rr = [p["r"] / p["r_star"] for p in res["points"] if p["r_star"]]
        VV = [p["V"] for p in res["points"] if p["r_star"]]
        ax.scatter(rr, VV, s=22, alpha=0.8, label=res["trace"])
    ax.set_xlabel("r / r*  (real within-epoch drift, normalised)")
    ax.set_ylabel("V (measured on real trace)")
    ax.set_xlim(0, 2.5); ax.set_ylim(0, 1.6)
    ax.set_title("H2c: adaptivity-value law on real traces"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "real_frontier.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
