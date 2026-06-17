# lsm-bloom-allocation — when is adaptive LSM Bloom tuning worth it?

A pre-registered study of **the value of adaptivity** in LSM-tree Bloom-filter allocation. We don't
propose a new filter or a new system; we characterise *when* the elaborate machinery for online,
workload-adaptive bit allocation actually pays — from first principles, validated on synthetic sweeps,
**real Twitter production cache traces**, and a **real RocksDB engine**.

> Problem: [`dbms_research/11-nosql-kv/adaptive-bloom-filter-tuning`](https://github.com/samyama-ai/dbms_research).
> This is an honest baseline-and-characterisation (mode b), not a SOTA claim. Limitations are in §6 and the paper.
>
> **Preprint:** [arXiv:2606.18138](https://arxiv.org/abs/2606.18138) (cs.DB).

## The model

An LSM run `i` holds `N_i` keys with a Bloom filter at `b_i` bits/key (FPR `f_i = e^{-β b_i}`,
`β = ln²2`). A negative lookup wastes one I/O per filter false-positive, so expected wasted I/O is
`C = Σ_i a_i f_i` (`a_i` = how often run/segment `i` is probed), minimised at fixed budget `Σ_i N_i b_i ≤ M`.
The static optimum (Monkey, Dayan SIGMOD'17) is `f_i* = λ N_i / a_i`.

## Three results

1. **The log-law.** Rewriting the optimum, **optimal bits-per-key is affine in `log(access frequency)`**,
   slope exactly `1/β = 2.081`. (Verified R² = 1.000.) The workload enters *only through its logarithm*.

2. **The robustness law.** Because the signal is logarithmic, allocation is strikingly insensitive to
   hotness-estimation error: **excess read cost = `D/2`** where `D` is the *N-weighted variance of the
   log misestimate* (slope ½, R² = 0.999), and a **common-factor error is fully absorbed** by the budget
   multiplier λ (≈0% excess). You need only a *coarse* hotness estimate to capture nearly all the benefit.

3. **The adaptivity-value frontier (two clocks).** Compaction rebuilds filters *for free* on its own
   clock. So the value `V` of continuous online tracking over a coarse log-law allocation **recomputed
   only at compaction** obeys a small-drift law `V ≈ ½ (r/r*)²`, where `r` is the within-epoch
   differential log-drift and **`r* = √(2 (C_uniform − C_oracle)/C_oracle)`**. Three regimes:
   - `r ≲ 0.34 r*`: **compaction-only keeps ≥90%** of adaptation's benefit at zero extra write cost.
   - middle: online tracking progressively pays.
   - `r ≳ 4 r*`: a *stale* concentrated allocation is **worse than uniform** (`V>1`) → fall back to uniform.
   More workload skew ⇒ larger `r*` ⇒ fine tracking matters **even less**.

## Results table

| Test | Claim | Result | Status |
|---|---|---|---|
| H0 | log-law slope = 1/β | slope **2.081**, R² **1.000**; Monkey −≥30% vs uniform | ✅ |
| H1 | excess = D/2, common-factor-immune | slope **0.50**, R² **0.999**; common/diff ratio ≈ **0** | ✅ |
| H2 | small-drift law `V≈½(r/r*)²` | c=**0.44**, R² **0.96**; coarse-suffices `r≤0.34 r*`; stale>uniform at `r≥4.2 r*` | ✅ |
| H2c | law transfers to real traces | regime-dependent: global MAE 0.23 (pre-reg test fails), **in-regime MAE 0.015**; 3 regimes confirmed on real Twitter clusters (one shows compaction-only = **96–99%** of tracking) | ⚠️ honest |
| H3 | FPR `=e^{-β b}` on real RocksDB | β_eff **0.462** vs 0.480 (bits≤8, within 4%); saturates 7.7× at 16 bits (Ribbon territory) | ✅ |

## Reproduce
```
pip install -e . && pytest -q          # 26 tests
python bench/run_synthetic.py          # H0/H1/H2 + figures
# real data:
bench/fetch_data.sh ./data 110000000 45 34 22
python bench/run_traces.py --data-dir ./data --fmt twitter   # H2c
sudo apt-get install -y rocksdb-tools && bench/rocksdb_fpr.sh # H3
```
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and the frozen [PREREGISTRATION.md](PREREGISTRATION.md).

## §6 Limitations & honest scope
- We characterise a *policy frontier*; we do **not** ship a new filter or a RocksDB fork. Per-segment
  *dynamic* reallocation in a real engine (ElasticBF territory) is left to future work; our RocksDB arm
  validates the FPR primitive and the Monkey direction, not a full per-segment controller.
- The "continuous tracker" is idealised as a perfect oracle, which *favours* adaptation; real trackers
  sit above it, so our "coarse suffices" conclusion is conservative.
- The model FPR `e^{-β b}` is the optimal-Bloom floor; real RocksDB sits a constant above it and
  saturates at high bits/key (use Ribbon there).
- "Robustness to estimation error" is thematically related to our sibling cardinality-estimation work,
  but the mechanism here (logarithmic signal dependence intrinsic to the bit→FPR curve) and the
  deliverable (clock-governed adaptivity frontier) are specific to AMQ filter allocation. See the paper's
  related-work and novelty discussion.

## License
Apache-2.0 (code). Builds on Monkey, ElasticBF, Endure, Ribbon — cited in the paper.
