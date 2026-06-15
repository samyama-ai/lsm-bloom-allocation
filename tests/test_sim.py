"""LSM-model + simulator tests, incl. the negative controls (NC1/NC2) from HYPOTHESIS.md."""
import numpy as np
import pytest

from lsm_bloom import (
    level_capacities, negative_access_weights, monkey_allocation, cost,
    uniform_bits, simulate, LogDriftWorkload,
)


def test_level_capacities_sum_and_shape():
    N = 1e6
    caps = level_capacities(N, T=10, L=5)
    assert caps.sum() == pytest.approx(N)
    assert np.all(np.diff(caps) > 0)            # deeper levels strictly larger
    assert caps[-1] / N > 0.8                   # deepest holds ~(T-1)/T of data


def test_negative_access_weights_normalised():
    caps = level_capacities(1e5, 10, 4)
    a = negative_access_weights(caps, mode="uniform_key")
    assert a.sum() == pytest.approx(1.0)
    given = negative_access_weights(caps, key_access=np.array([4, 3, 2, 1.0]), mode="given")
    assert given.sum() == pytest.approx(1.0)


def test_uniform_is_optimal_when_access_tracks_capacity():
    # NC-flavoured: if a_i ∝ N_i then f_i ∝ N_i/a_i is constant => uniform bits/key IS optimal.
    caps = level_capacities(1e5, 10, 4)
    a = negative_access_weights(caps, mode="uniform_key")   # a_i ∝ N_i
    M = 8.0 * caps.sum()
    b_opt = monkey_allocation(caps, a, M)
    assert np.allclose(b_opt, b_opt[0], rtol=1e-6)          # equal bits/key
    assert cost(a, b_opt) == pytest.approx(cost(a, uniform_bits(caps, M)), rel=1e-6)


def test_monkey_beats_uniform_under_skew():
    # H0a direction: skewed access (not ∝ N) => Monkey strictly cheaper than uniform.
    caps = level_capacities(1e6, 10, 5)
    a = negative_access_weights(caps, key_access=np.array([100, 50, 10, 2, 1.0]), mode="given")
    M = 8.0 * caps.sum()
    c_uniform = cost(a, uniform_bits(caps, M))
    c_monkey = cost(a, monkey_allocation(caps, a, M))
    assert c_monkey < c_uniform


def test_nc1_stationary_no_adaptivity_value():
    # NC1: zero drift => compaction-only == oracle => V ~ 0 (no harness leak).
    caps = level_capacities(1e5, 10, 4)
    wl = LogDriftWorkload(R=4, sigma=0.0, epoch=20, s=0.99, seed=1)
    out = simulate(caps, wl, M=8.0 * caps.sum(), steps=400, warmup=40)
    assert out["r"] == pytest.approx(0.0, abs=1e-9)
    assert out["V"] < 0.03


def test_nc2_budget_monotonic():
    # NC2: optimal cost is non-increasing in memory budget M.
    caps = level_capacities(1e5, 10, 4)
    a = negative_access_weights(caps, key_access=np.array([10, 5, 2, 1.0]), mode="given")
    costs = [cost(a, monkey_allocation(caps, a, m * caps.sum())) for m in (2, 4, 8, 12, 16)]
    assert all(costs[i + 1] <= costs[i] + 1e-12 for i in range(len(costs) - 1))


def test_drift_increases_adaptivity_value():
    # Monotone direction for H2a: more within-epoch drift => larger V.
    caps = level_capacities(1e5, 10, 4)
    Vs = []
    for sigma in (0.01, 0.05, 0.2):
        wl = LogDriftWorkload(R=4, sigma=sigma, epoch=30, s=0.99, seed=2)
        Vs.append(simulate(caps, wl, M=8.0 * caps.sum(), steps=900, warmup=60)["V"])
    assert Vs[0] < Vs[2]
    assert Vs[2] > Vs[1] - 0.05


def test_compaction_only_between_static_and_oracle():
    caps = level_capacities(1e5, 10, 4)
    wl = LogDriftWorkload(R=4, sigma=0.1, epoch=25, s=0.99, seed=4)
    out = simulate(caps, wl, M=8.0 * caps.sum(), steps=800, warmup=50)
    assert out["C_oracle"] <= out["C_compact_only"] + 1e-9
    assert out["C_compact_only"] <= out["C_static"] + 1e-9   # refreshing helps vs frozen
