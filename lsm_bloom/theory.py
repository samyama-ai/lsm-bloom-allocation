"""Closed-form theory for LSM Bloom-filter allocation.

All math here is pre-registered in dbms_cloud/daily/11-adaptive-bloom-filter-tuning/HYPOTHESIS.md.

Core objects
------------
- Bloom filter (optimal #hashes):  f = exp(-BETA * b),  b = bits/key,  BETA = ln(2)^2.
- Objective (expected wasted I/O per *negative* lookup):  C(b) = sum_i a_i * exp(-BETA*b_i).
- Budget:  sum_i N_i * b_i <= M  (M = total filter bits).
- Static optimum (Monkey, Dayan SIGMOD'17):  f_i* = lambda * N_i / a_i,  i.e.
  b_i* = (ln a_i - ln N_i - ln lambda) / BETA   -> bits/key AFFINE in ln(access frequency).

The 'log-law' (slope 1/BETA in ln a_i) and its second-order robustness corollary
(excess cost ~ N-weighted variance of log misestimate) are the levers; see HYPOTHESIS.md.
"""
from __future__ import annotations

import numpy as np

BETA = np.log(2.0) ** 2          # ln(2)^2 ~= 0.4804530139182014
LOG_LAW_SLOPE = 1.0 / BETA       # ~= 2.0813689810056077  (bits/key per nat of ln a_i)


# ----------------------------------------------------------------------------- Bloom primitives
def bloom_fpr(bits_per_key: np.ndarray | float) -> np.ndarray | float:
    """False-positive rate of an optimally-tuned Bloom filter at b bits/key: exp(-BETA*b)."""
    return np.exp(-BETA * np.asarray(bits_per_key, dtype=float))


def bits_for_fpr(f: np.ndarray | float) -> np.ndarray | float:
    """Bits/key needed for FPR f: ln(1/f)/BETA.  (Inverse of bloom_fpr.)"""
    f = np.asarray(f, dtype=float)
    return -np.log(f) / BETA


def cost(a: np.ndarray, b: np.ndarray) -> float:
    """Expected wasted I/O per negative lookup: sum_i a_i * fpr(b_i)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sum(a * bloom_fpr(b)))


# ----------------------------------------------------------------------- Static optimum (Monkey)
def monkey_allocation(N: np.ndarray, a: np.ndarray, M: float) -> np.ndarray:
    """Optimal bits/key per run minimising C(b)=sum a_i exp(-BETA b_i) s.t. sum N_i b_i = M, b_i>=0.

    Water-filling with a non-negativity floor (a run can receive 0 bits => no filter, FPR=1).
    Closed form per active set S:  ln(lambda) = [ sum_{i in S} N_i (ln a_i - ln N_i) - BETA*M ] / sum_{i in S} N_i,
    then b_i = (ln a_i - ln N_i - ln lambda)/BETA.  Runs with b_i<0 are dropped (b_i:=0) and we re-solve.
    """
    N = np.asarray(N, dtype=float)
    a = np.asarray(a, dtype=float)
    if np.any(N <= 0):
        raise ValueError("N_i must be positive")
    if np.any(a < 0):
        raise ValueError("a_i must be non-negative")
    if M < 0:
        raise ValueError("M must be non-negative")

    active = a > 0                      # runs never probed get no bits
    while True:
        idx = np.where(active)[0]
        if idx.size == 0:
            return np.zeros_like(N)
        Ns, as_ = N[idx], a[idx]
        ln_lambda = (np.sum(Ns * (np.log(as_) - np.log(Ns))) - BETA * M) / np.sum(Ns)
        b = (np.log(as_) - np.log(Ns) - ln_lambda) / BETA
        if np.all(b >= -1e-12):        # all active runs want non-negative bits -> done
            out = np.zeros_like(N)
            out[idx] = np.maximum(b, 0.0)
            return out
        active[idx[b < 0]] = False     # drop the most-starved runs, re-solve on the rest


def log_law_fit(a: np.ndarray, b: np.ndarray, N: np.ndarray | None = None) -> dict:
    """Fit b ~ slope*ln(a) + intercept on the unclipped (b>0) runs. Returns slope, intercept, R^2.

    Theory predicts slope == LOG_LAW_SLOPE (1/BETA). Optionally pass N to also report the
    N-controlled fit b ~ s1*ln(a) + s2*ln(N) (the exact form b=(ln a - ln N - ln lambda)/BETA)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = b > 1e-9
    x, y = np.log(a[mask]), b[mask]
    if x.size < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(x.size)}
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    out = {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n": int(x.size)}
    if N is not None:
        N = np.asarray(N, dtype=float)
        A2 = np.vstack([np.log(a[mask]), np.log(N[mask]), np.ones(mask.sum())]).T
        coef, *_ = np.linalg.lstsq(A2, y, rcond=None)
        out["slope_lna"], out["slope_lnN"] = float(coef[0]), float(coef[1])
    return out


# ------------------------------------------------------------------- Robustness law (H1)
def nweighted_logvar(kappa: np.ndarray, N: np.ndarray) -> float:
    """N-weighted variance of the log misestimate: D = sum w_i (ln k_i - <ln k>_w)^2, w_i=N_i/sum N.

    This is the order parameter of the robustness law (H1): excess cost is predicted ~ linear in D,
    and a common-factor error (all k_i equal => D=0) is absorbed by lambda-rebalance (~0 penalty)."""
    kappa = np.asarray(kappa, dtype=float)
    N = np.asarray(N, dtype=float)
    w = N / np.sum(N)
    lk = np.log(kappa)
    mean = np.sum(w * lk)
    return float(np.sum(w * (lk - mean) ** 2))


def excess_cost_fraction(N: np.ndarray, a_true: np.ndarray, kappa: np.ndarray, M: float) -> float:
    """Relative excess cost from allocating with misestimated hotness a_hat = kappa * a_true.

    (C(b(a_hat); a_true) - C(b(a_true); a_true)) / C(b(a_true); a_true).  >= 0 by optimality."""
    a_true = np.asarray(a_true, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    b_opt = monkey_allocation(N, a_true, M)
    b_mis = monkey_allocation(N, a_true * kappa, M)
    c_opt = cost(a_true, b_opt)
    c_mis = cost(a_true, b_mis)
    return (c_mis - c_opt) / c_opt if c_opt > 0 else np.nan


# ------------------------------------------------------- Adaptivity-value frontier (H2 closed form)
def adaptivity_value_closed_form(r: np.ndarray | float, r_star: float) -> np.ndarray | float:
    """Predicted value of continuous tracking over compaction-only reallocation: 1 - exp(-(r/r*)^2).

    r = sqrt(within-epoch differential N-weighted log-drift variance) (HYPOTHESIS.md H2b).
    Rises from 0 (coarse-at-compaction suffices) to 1 (fine tracking essential) at scale r*."""
    r = np.asarray(r, dtype=float)
    return 1.0 - np.exp(-(r / r_star) ** 2)
