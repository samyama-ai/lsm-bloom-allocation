"""Workloads: skewed access weights and a controllable drift process.

The simulator treats the corpus as R 'regions' (levels, or per-SSTable segments), each with a
time-varying negative-access weight a_i(t). Drift is a multiplicative log-random-walk whose
*differential* magnitude (the only thing the robustness/adaptivity laws care about) is tunable.
"""
from __future__ import annotations

import numpy as np


def zipf_weights(R: int, s: float, rng: np.random.Generator | None = None,
                 shuffle: bool = True) -> np.ndarray:
    """Zipfian access weights over R regions, exponent s. Normalised to sum 1."""
    ranks = np.arange(1, R + 1, dtype=float)
    w = 1.0 / ranks ** s
    w = w / w.sum()
    if shuffle and rng is not None:
        w = w[rng.permutation(R)]
    return w


class LogDriftWorkload:
    """Multiplicative log-random-walk on region access weights.

    a_i(t+1) = normalise( a_i(t) * exp(eps_i) ), eps_i ~ N(0, sigma^2) i.i.d. across regions/steps.
    sigma sets the per-step *differential* log-drift; over a compaction epoch of E steps the
    accumulated differential log-drift std is ~ sigma*sqrt(E). Mean-reversion (theta>0) keeps the
    walk stationary-ish so long runs don't degenerate to one hot region.
    """

    def __init__(self, R: int, sigma: float, epoch: int, s: float = 0.99,
                 theta: float = 0.0, seed: int = 0):
        self.R = R
        self.sigma = float(sigma)
        self.epoch = int(epoch)
        self.theta = float(theta)
        self.rng = np.random.default_rng(seed)
        self.log_base = np.log(zipf_weights(R, s, self.rng))
        self.log_a = self.log_base.copy()
        self.t = 0

    def _normalise(self, log_a: np.ndarray) -> np.ndarray:
        a = np.exp(log_a - log_a.max())
        return a / a.sum()

    def current(self) -> np.ndarray:
        return self._normalise(self.log_a)

    def step(self) -> np.ndarray:
        eps = self.rng.normal(0.0, self.sigma, self.R)
        # Ornstein-Uhlenbeck-style pull toward the base profile (keeps it stationary for long sims)
        self.log_a = self.log_a + eps - self.theta * (self.log_a - self.log_base)
        self.t += 1
        return self.current()

    def is_compaction(self) -> bool:
        """Compaction (free filter rebuild) fires every `epoch` steps."""
        return self.t > 0 and (self.t % self.epoch == 0)


def nweighted_logdrift(a_start: np.ndarray, a_end: np.ndarray, N: np.ndarray) -> float:
    """sqrt of N-weighted variance of ln(a_end/a_start): the drift ratio r in HYPOTHESIS.md H2."""
    a_start = np.asarray(a_start, dtype=float)
    a_end = np.asarray(a_end, dtype=float)
    N = np.asarray(N, dtype=float)
    w = N / N.sum()
    d = np.log(a_end) - np.log(a_start)
    mean = np.sum(w * d)
    return float(np.sqrt(np.sum(w * (d - mean) ** 2)))
