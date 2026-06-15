"""LSM-tree structural model: run capacities and negative-access weights.

A leveled LSM with size ratio T and L levels: level i (1=shallowest) holds ~ N*(T-1)/T^(L+1-i)
keys, so the deepest level holds ~ (T-1)/T of all data. We model one run per level (leveling).
"""
from __future__ import annotations

import numpy as np


def level_capacities(N: float, T: int, L: int) -> np.ndarray:
    """Keys per level for a leveled LSM of N keys, size ratio T, L levels (shallow->deep).

    Level i (1-indexed) capacity ~ N*(T-1)/T^(L+1-i). Normalised so the sum is exactly N."""
    i = np.arange(1, L + 1)
    cap = (T - 1.0) / T ** (L + 1 - i)
    cap = cap / cap.sum() * N
    return cap


def num_levels(N: float, T: int, entries_per_buffer: float = 1.0) -> int:
    """Standard LSM depth L = ceil(log_T(N/buffer * (T-1)/T)). Kept simple; buffer in 'entries'."""
    return max(1, int(np.ceil(np.log(N / entries_per_buffer) / np.log(T))))


def negative_access_weights(caps: np.ndarray, key_access: np.ndarray | None = None,
                            mode: str = "uniform_key") -> np.ndarray:
    """Negative-access weight a_i: probability a negative lookup probes run i.

    A negative (absent-key) lookup probes filters top-down until one returns 'maybe'. In the
    *no-false-positive-above* idealisation every negative lookup that isn't short-circuited reaches
    every level, so the access weight is driven by which keys are queried, not by the LSM order.

    mode='uniform_key':  a_i proportional to caps (keys spread uniformly) -> recovers RocksDB's
                         equal-FPR default as the optimum (sanity case).
    mode='given':        a_i proportional to a supplied per-level key_access mass (skewed workload).
    """
    caps = np.asarray(caps, dtype=float)
    if mode == "uniform_key":
        a = caps.copy()
    elif mode == "given":
        if key_access is None:
            raise ValueError("mode='given' requires key_access")
        a = np.asarray(key_access, dtype=float)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    s = a.sum()
    return a / s if s > 0 else a
