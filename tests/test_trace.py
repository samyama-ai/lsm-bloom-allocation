"""Parser-correctness tests for trace processing (real-format fixtures, not result mocks).

These verify the pipeline turns a known trace into the right segment-hotness matrix; the actual
H2c numbers come from REAL downloaded traces via bench/run_traces.py.
"""
import numpy as np

from lsm_bloom import trace as TR
from lsm_bloom.policies import simulate_sequence


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def test_twitter_reads_only_gets(tmp_path):
    lines = [
        "0,keyA,10,100,1,get,0",
        "1,keyB,10,100,1,set,0",     # writes ignored
        "2,keyA,10,100,1,gets,0",
        "3,keyC,10,100,1,get,0",
    ]
    p = _write(tmp_path, "c.csv", lines)
    keys = list(TR.iter_keys(p, "twitter"))
    assert keys == ["keyA", "keyA", "keyC"]      # the 'set' is dropped


def test_wiki_expands_counts(tmp_path):
    lines = ["en Foo 3 0", "en Bar 1 0"]
    p = _write(tmp_path, "pageviews-x", lines)
    keys = list(TR.iter_keys(p, "wiki"))
    assert keys.count("en/Foo") == 3 and keys.count("en/Bar") == 1


def test_sequence_shape_and_normalised(tmp_path):
    # 600 kv accesses, window 100 -> 6 windows over R=8 segments, each row sums to 1.
    rng = np.random.default_rng(0)
    lines = [f"{i},key{rng.integers(0, 50)}" for i in range(600)]
    p = _write(tmp_path, "t.txt", lines)
    a_seq = TR.trace_to_sequence(p, "kv", R=8, window=100, max_windows=50)
    assert a_seq.shape == (6, 8)
    assert np.allclose(a_seq.sum(axis=1), 1.0)


def test_simulate_sequence_runs_on_real_format(tmp_path):
    # drifting access: first half hits low keys, second half high keys -> nonzero drift & V defined.
    lines = ([f"{i},key{i % 4}" for i in range(4000)] +
             [f"{i},key{8 + i % 4}" for i in range(4000)])
    p = _write(tmp_path, "drift.txt", lines)
    a_seq = TR.trace_to_sequence(p, "kv", R=16, window=400, max_windows=50)
    N = np.full(16, 1e5 / 16)
    out = simulate_sequence(N, a_seq, compaction_period=4, M=10.0 * N.sum(), warmup=2)
    assert out["n_windows"] >= 10
    assert out["C_oracle"] <= out["C_compact_only"] + 1e-9
    assert np.isfinite(out["r"]) and out["r"] >= 0
