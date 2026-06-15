"""Allocation policies and the drift simulator that measures the adaptivity-value frontier.

Policies (all share the SAME total budget M):
- uniform        : equal bits/key everywhere (RocksDB default family). Never updates.
- static_optimal : Monkey allocation from the initial hotness. Never updates.
- compaction_only: Monkey allocation recomputed ONLY at compaction events using the hotness observed
                   then (the 'free rebuild' clock); frozen between events. <-- our proposed coarse policy.
- oracle         : Monkey allocation recomputed every step from the true current hotness (lower bound).
- continuous     : idealised perfect tracker == oracle (ElasticBF's continuous adaptation, best case).
                   Real trackers sit ABOVE oracle (estimation noise), so this is a conservative,
                   adaptation-favouring choice: it makes 'continuous' look as good as possible.
"""
from __future__ import annotations

import numpy as np

from .theory import cost, monkey_allocation
from .workload import LogDriftWorkload, nweighted_logdrift


def uniform_bits(N: np.ndarray, M: float) -> np.ndarray:
    """Equal bits/key b = M / sum(N) for every run."""
    N = np.asarray(N, dtype=float)
    return np.full_like(N, M / N.sum())


def simulate(N: np.ndarray, wl: LogDriftWorkload, M: float, steps: int,
             warmup: int = 0) -> dict:
    """Run the drift workload `steps` steps; return time-averaged cost of each policy and V.

    Returns dict with C_uniform, C_static, C_compact_only, C_oracle (== C_continuous),
    V (adaptivity value), and mean per-epoch drift r (sqrt N-weighted differential log-drift var).
    """
    N = np.asarray(N, dtype=float)
    b_uniform = uniform_bits(N, M)

    a0 = wl.current()
    b_static = monkey_allocation(N, a0, M)
    b_compact = b_static.copy()           # refreshed at each compaction event
    a_epoch_start = a0.copy()

    acc = {"uniform": 0.0, "static": 0.0, "compact": 0.0, "oracle": 0.0}
    drift_samples = []
    n_counted = 0

    for t in range(steps):
        a = wl.step()
        if wl.is_compaction():
            drift_samples.append(nweighted_logdrift(a_epoch_start, a, N))
            b_compact = monkey_allocation(N, a, M)   # free rebuild uses current hotness
            a_epoch_start = a.copy()
        if t < warmup:
            continue
        b_oracle = monkey_allocation(N, a, M)
        acc["uniform"] += cost(a, b_uniform)
        acc["static"] += cost(a, b_static)
        acc["compact"] += cost(a, b_compact)
        acc["oracle"] += cost(a, b_oracle)
        n_counted += 1

    for k in acc:
        acc[k] /= max(1, n_counted)

    gap = acc["uniform"] - acc["oracle"]
    V = (acc["compact"] - acc["oracle"]) / gap if gap > 1e-15 else 0.0
    return {
        "C_uniform": acc["uniform"],
        "C_static": acc["static"],
        "C_compact_only": acc["compact"],
        "C_oracle": acc["oracle"],
        "C_continuous": acc["oracle"],          # idealised perfect tracker
        "V": float(np.clip(V, 0.0, 1.5)),
        "r": float(np.mean(drift_samples)) if drift_samples else 0.0,
        "n_epochs": len(drift_samples),
    }


def simulate_sequence(N: np.ndarray, a_seq: np.ndarray, compaction_period: int, M: float,
                      warmup: int = 0) -> dict:
    """Drive the policies with an EXPLICIT per-window access-weight sequence a_seq[t] (real traces).

    a_seq: shape (T_windows, R), each row a probability vector over R segments. Compaction (free
    rebuild) fires every `compaction_period` windows. Returns the same keys as simulate(), with the
    measured drift r = mean sqrt(N-weighted differential log-drift) per compaction epoch.
    """
    N = np.asarray(N, dtype=float)
    a_seq = np.asarray(a_seq, dtype=float)
    a_seq = np.clip(a_seq, 1e-12, None)
    a_seq = a_seq / a_seq.sum(axis=1, keepdims=True)
    Tw, R = a_seq.shape
    b_uniform = uniform_bits(N, M)
    b_static = monkey_allocation(N, a_seq[0], M)
    b_compact = b_static.copy()
    a_epoch_start = a_seq[0].copy()

    acc = {"uniform": 0.0, "static": 0.0, "compact": 0.0, "oracle": 0.0}
    drift, n = [], 0
    for t in range(Tw):
        a = a_seq[t]
        if t > 0 and t % compaction_period == 0:
            drift.append(nweighted_logdrift(a_epoch_start, a, N))
            b_compact = monkey_allocation(N, a, M)
            a_epoch_start = a.copy()
        if t < warmup:
            continue
        b_oracle = monkey_allocation(N, a, M)
        acc["uniform"] += cost(a, b_uniform)
        acc["static"] += cost(a, b_static)
        acc["compact"] += cost(a, b_compact)
        acc["oracle"] += cost(a, b_oracle)
        n += 1
    for k in acc:
        acc[k] /= max(1, n)
    gap = acc["uniform"] - acc["oracle"]
    V = (acc["compact"] - acc["oracle"]) / gap if gap > 1e-15 else 0.0
    return {
        "C_uniform": acc["uniform"], "C_static": acc["static"],
        "C_compact_only": acc["compact"], "C_oracle": acc["oracle"], "C_continuous": acc["oracle"],
        "V": float(np.clip(V, 0.0, 2.0)), "r": float(np.mean(drift)) if drift else 0.0,
        "r_star_theory": float(np.sqrt(2 * gap / acc["oracle"])) if acc["oracle"] > 0 else float("nan"),
        "n_epochs": len(drift), "n_windows": int(Tw),
    }
