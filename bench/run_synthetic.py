"""One-command synthetic run: H0 (gate), H1 (robustness), H2 (adaptivity frontier).

    python bench/run_synthetic.py
Writes results/{H0,H1,H2}.json and figures/{loglaw,robustness,frontier}.png.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsm_bloom import experiment as E         # noqa: E402
from lsm_bloom import theory as T             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def _save(name, obj):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=2)


def main():
    h0 = E.run_H0()
    h1 = E.run_H1()
    h2 = E.run_H2()
    _save("H0.json", h0)
    _save("H1.json", h1)
    _save("H2.json", h2)

    print("== H0 gate ==")
    print(f"  Monkey vs uniform reduction: {h0['reduction']*100:.1f}%  (>=30%? {h0['H0a_pass']})")
    print(f"  log-law slope: {h0['log_law_slope']:.4f} vs theory {h0['log_law_slope_theory']:.4f} "
          f"(R2={h0['log_law_r2']:.4f})  pass={h0['H0b_pass']}")
    print("== H1 robustness law ==")
    print(f"  excess ~ slope*D : R2={h1['linear_r2']:.4f}  (quadratic-in-log? {h1['H1a_pass']})")
    print(f"  common/differential excess ratio: {h1['common_to_differential_ratio']:.4f}  "
          f"(common-factor immune? {h1['H1b_pass']})")
    print("== H2 adaptivity-value frontier ==")
    print(f"  r*_theory=sqrt(2 gap/C_oracle) = {h2['r_star_theory']:.4f}")
    print(f"  small-drift law V≈c*(r/r*)^2 : c={h2['quad_c']:.3f} (theory 0.5)  R2={h2['quad_r2']:.4f}")
    print(f"  coarse-suffices boundary r(V<=0.1) = {h2['r_boundary_V0.1']:.3f} "
          f"({h2['boundary_over_rstar']:.2f} x r*) | V_max={h2['V_max']:.3f}")
    print(f"  H2a (regime exists) pass={h2['H2a_pass']} | H2b (small-drift law) pass={h2['H2b_pass']}")

    _plot(h0, h1, h2)
    print(f"\nfigures -> {FIG}")


def _plot(h0, h1, h2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # log-law
    rng = np.random.default_rng(0)
    Neq = np.full(50, 1000.0)
    a = rng.dirichlet(np.ones(50) * 0.3)
    b = T.monkey_allocation(Neq, a, 8.0 * Neq.sum())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(np.log(a), b, s=14, alpha=0.7)
    xs = np.linspace(np.log(a).min(), np.log(a).max(), 50)
    ax.plot(xs, T.LOG_LAW_SLOPE * xs + (b.mean() - T.LOG_LAW_SLOPE * np.log(a).mean()),
            "r--", label=f"slope 1/ln^2 2 = {T.LOG_LAW_SLOPE:.3f}")
    ax.set_xlabel("ln(access frequency a_i)"); ax.set_ylabel("optimal bits/key b_i*")
    ax.set_title("Log-law: bits/key affine in log-hotness"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "loglaw.png"), dpi=130); plt.close(fig)

    # robustness
    fig, ax = plt.subplots(figsize=(5, 4))
    D, ex = np.array(h1["D"]), np.array(h1["excess"])
    ax.scatter(D, ex * 100, s=10, alpha=0.5)
    xs = np.linspace(0, D.max(), 50)
    ax.plot(xs, h1["linear_slope"] * xs * 100, "r--",
            label=f"linear in D (R2={h1['linear_r2']:.3f})")
    ax.set_xlabel("D = N-weighted var(ln hotness-error)")
    ax.set_ylabel("excess read cost (%)")
    ax.set_title("Robustness: excess cost quadratic in log-error"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "robustness.png"), dpi=130); plt.close(fig)

    # frontier
    fig, ax = plt.subplots(figsize=(5, 4))
    r, V = np.array(h2["r"]), np.array(h2["V"])
    rstar = h2["r_star_theory"]
    ax.scatter(r, V, s=22, label="simulated", zorder=3)
    xs = np.linspace(0, min(r.max(), rstar), 60)
    ax.plot(xs, h2["quad_c"] * (xs / rstar) ** 2, "r--",
            label=f"small-drift law c(r/r*)^2, c={h2['quad_c']:.2f}")
    ax.axvline(h2["r_boundary_V0.1"], color="green", ls=":", alpha=0.7,
               label=f"coarse-suffices r(V<=.1)={h2['r_boundary_V0.1']:.2f}")
    ax.axhline(0.1, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("r = within-epoch differential log-drift")
    ax.set_ylabel("V = value of continuous tracking")
    ax.set_title("Adaptivity-value frontier (two clocks)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "frontier.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
