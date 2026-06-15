"""Correctness tests for the closed-form theory (test layers 2 + 5 from HYPOTHESIS.md stage 3)."""
import numpy as np
import pytest
from scipy.optimize import minimize

from lsm_bloom import theory as T


def test_bloom_fpr_inverse():
    b = np.array([2.0, 5.0, 10.0, 14.0])
    assert np.allclose(T.bits_for_fpr(T.bloom_fpr(b)), b)
    # 10 bits/key -> ~0.8% FPR (textbook optimal-k Bloom)
    assert 0.006 < T.bloom_fpr(10.0) < 0.010


def test_log_law_slope_constant():
    assert T.LOG_LAW_SLOPE == pytest.approx(1.0 / (np.log(2) ** 2))
    assert T.LOG_LAW_SLOPE == pytest.approx(2.0813689810, abs=1e-6)


def test_monkey_budget_exact():
    rng = np.random.default_rng(0)
    N = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    a = rng.dirichlet(np.ones(5))
    M = 6.0 * N.sum()
    b = T.monkey_allocation(N, a, M)
    assert np.all(b >= -1e-12)
    assert np.sum(N * b) == pytest.approx(M, rel=1e-9)   # uses full budget when interior


def test_monkey_matches_numerical_optimum():
    # Compare the closed-form water-filling allocation to a generic constrained minimiser.
    N = np.array([1.0, 8.0, 64.0, 512.0])
    a = np.array([0.4, 0.3, 0.2, 0.1])
    M = 5.0 * N.sum()
    b_cf = T.monkey_allocation(N, a, M)

    def obj(b):
        return T.cost(a, b)
    cons = {"type": "eq", "fun": lambda b: np.sum(N * b) - M}
    bounds = [(0, None)] * len(N)
    res = minimize(obj, x0=np.full(len(N), M / N.sum()), bounds=bounds,
                   constraints=cons, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    assert res.success
    assert T.cost(a, b_cf) <= T.cost(a, res.x) + 1e-9      # closed form no worse than numeric
    assert T.cost(a, b_cf) == pytest.approx(T.cost(a, res.x), rel=1e-4)


def test_monkey_shape_deep_levels_higher_fpr():
    # f_i ∝ N_i/a_i : with equal access, the larger (deeper) level gets the HIGHER FPR.
    N = np.array([10.0, 1000.0])
    a = np.array([0.5, 0.5])
    b = T.monkey_allocation(N, a, 6.0 * N.sum())
    f = T.bloom_fpr(b)
    assert f[1] > f[0]                                     # deep level less protected
    assert b[1] < b[0]                                     # fewer bits/key deep


def test_log_law_slope_equal_N():
    # Equal N isolates the ln(a) slope: b vs ln(a) is affine with slope exactly 1/BETA.
    rng = np.random.default_rng(3)
    R = 40
    N = np.full(R, 100.0)
    a = rng.dirichlet(np.ones(R) * 0.3)
    b = T.monkey_allocation(N, a, 8.0 * N.sum())
    fit = T.log_law_fit(a, b)
    assert fit["slope"] == pytest.approx(T.LOG_LAW_SLOPE, rel=0.05)
    assert fit["r2"] > 0.999            # exactly linear when no run hits the b>=0 floor


def test_log_law_both_slopes_varying_N():
    # Varying N identifies both regressors: b = (ln a - ln N - ln lambda)/BETA.
    rng = np.random.default_rng(11)
    R = 60
    N = np.exp(rng.uniform(0, 8, R))                       # spread N over orders of magnitude
    a = rng.dirichlet(np.ones(R) * 0.3)
    b = T.monkey_allocation(N, a, 10.0 * N.sum())
    fit = T.log_law_fit(a, b, N=N)
    assert fit["slope_lna"] == pytest.approx(T.LOG_LAW_SLOPE, rel=0.05)
    assert fit["slope_lnN"] == pytest.approx(-T.LOG_LAW_SLOPE, rel=0.05)


def test_excess_cost_nonnegative_and_zero_at_truth():
    N = np.array([1.0, 10.0, 100.0, 1000.0])
    a = np.array([0.4, 0.3, 0.2, 0.1])
    M = 6.0 * N.sum()
    assert T.excess_cost_fraction(N, a, np.ones(4), M) == pytest.approx(0.0, abs=1e-9)
    bad = np.array([4.0, 0.25, 4.0, 0.25])
    assert T.excess_cost_fraction(N, a, bad, M) > 0.0


def test_common_factor_misestimate_is_absorbed():
    # H1b: a uniform (common-factor) hotness error => D=0 => ~zero excess cost (lambda rebalances).
    N = np.array([1.0, 10.0, 100.0, 1000.0, 5000.0])
    a = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    M = 6.0 * N.sum()
    for kappa in (0.5, 2.0, 5.0):
        common = np.full(5, kappa)
        assert T.nweighted_logvar(common, N) == pytest.approx(0.0, abs=1e-12)
        assert T.excess_cost_fraction(N, a, common, M) == pytest.approx(0.0, abs=1e-6)


def test_excess_cost_quadratic_in_logvar():
    # H1a (unit-scale): excess cost grows ~ linearly in D = N-weighted log-variance of the error.
    rng = np.random.default_rng(7)
    N = np.array([1.0, 10.0, 100.0, 1000.0])
    a = np.array([0.4, 0.3, 0.2, 0.1])
    M = 7.0 * N.sum()
    Ds, Es = [], []
    for scale in np.linspace(0.05, 0.7, 12):
        lk = rng.normal(0, scale, 4)
        lk -= lk.mean()
        kappa = np.exp(lk)
        Ds.append(T.nweighted_logvar(kappa, N))
        Es.append(T.excess_cost_fraction(N, a, kappa, M))
    Ds, Es = np.array(Ds), np.array(Es)
    # linear fit excess ~ slope*D through small D: strong correlation
    r = np.corrcoef(Ds, Es)[0, 1]
    assert r > 0.9


def test_adaptivity_value_monotone_bounds():
    r = np.linspace(0, 5, 50)
    V = T.adaptivity_value_closed_form(r, r_star=1.0)
    assert np.all(np.diff(V) >= -1e-12)
    assert V[0] == pytest.approx(0.0)
    assert V[-1] > 0.99
    assert T.adaptivity_value_closed_form(1.0, 1.0) == pytest.approx(1 - np.exp(-1), abs=1e-9)
