"""Smoke + gate tests for the experiment runners (reduced sizes for speed)."""
from lsm_bloom import experiment as E


def test_H0_gate_passes():
    h0 = E.run_H0()
    assert h0["H0a_pass"] and h0["H0b_pass"]
    assert h0["reduction"] > 0.30
    assert abs(h0["log_law_slope"] - h0["log_law_slope_theory"]) / h0["log_law_slope_theory"] < 0.10


def test_H1_robustness_law():
    h1 = E.run_H1(n_draws=200)
    assert h1["H1a_pass"]                       # excess ~ D/2, R2>=0.90, slope~0.5
    assert h1["H1b_pass"]                       # common-factor immune
    assert abs(h1["linear_slope"] - 0.5) < 0.1
    # the D/2 law is tight at small error, looser at large error (honest)
    assert h1["half_law_rel_err_small"] < h1["half_law_rel_err_large"]


def test_H2_adaptivity_frontier():
    # directional smoke (the strict H2a/H2b gates are validated in bench/run_synthetic.py)
    h2 = E.run_H2(sigmas=[0.01, 0.03, 0.06, 0.1, 0.2, 0.4, 0.8, 1.5, 2.5],
                  steps=700, warmup=70)
    assert 0.2 <= h2["quad_c"] <= 0.9           # small-drift law slope near 1/2
    assert h2["quad_r2"] >= 0.75
    assert h2["r_boundary_V0.1"] > 0            # a coarse-suffices regime exists
    assert h2["V_max"] >= 0.5                   # tracking eventually matters (stale->worse than uniform)
