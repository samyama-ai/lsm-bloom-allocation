"""lsm_bloom: analytical value-of-adaptivity for LSM Bloom-filter allocation.

Pre-registration: dbms_cloud/daily/11-adaptive-bloom-filter-tuning/HYPOTHESIS.md (frozen 2026-06-15).
"""
from .theory import (
    BETA, LOG_LAW_SLOPE,
    bloom_fpr, bits_for_fpr, cost, monkey_allocation, log_law_fit,
    nweighted_logvar, excess_cost_fraction, adaptivity_value_closed_form,
)
from .lsm import level_capacities, num_levels, negative_access_weights
from .workload import zipf_weights, LogDriftWorkload, nweighted_logdrift
from .policies import uniform_bits, simulate

__all__ = [
    "BETA", "LOG_LAW_SLOPE",
    "bloom_fpr", "bits_for_fpr", "cost", "monkey_allocation", "log_law_fit",
    "nweighted_logvar", "excess_cost_fraction", "adaptivity_value_closed_form",
    "level_capacities", "num_levels", "negative_access_weights",
    "zipf_weights", "LogDriftWorkload", "nweighted_logdrift",
    "uniform_bits", "simulate",
]
