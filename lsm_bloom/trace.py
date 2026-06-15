"""Real-workload trace processing for the H2c real-trace-transfer test.

Turns a key-access trace into a per-window segment-hotness sequence a_seq[t, i], then the
adaptivity-value frontier is evaluated on the REAL drift (lsm_bloom.policies.simulate_sequence).

Supported formats (auto by extension/flag):
- 'twitter'   : Twitter cache trace CSV  (timestamp,anon_key,key_size,value_size,client,op,ttl)
                https://github.com/twitter/cache-trace  (Yang et al., OSDI'20)
- 'wiki'      : Wikimedia pageview-complete lines  '<domain> <page> <count> <bytes>' per hourly file
                https://dumps.wikimedia.org/other/pageview_complete/
- 'kv'        : generic 'timestamp,key' or 'key' per line

No data is vendored; bench/fetch_data.sh downloads the real traces.
"""
from __future__ import annotations

import gzip
import zlib
from pathlib import Path

import numpy as np


def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def _seg(key: str, R: int) -> int:
    # crc32 is a fast, deterministic, uniform-enough bucketing of keys into R segments.
    return zlib.crc32(key.encode("utf-8", "ignore")) % R


def iter_keys(path, fmt: str):
    """Yield access keys (strings) in trace order. Counts are expanded for 'wiki'."""
    with _open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if fmt == "twitter":
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                op = parts[5].strip().lower()
                if op not in ("get", "gets"):       # reads only (negative-lookup proxy)
                    continue
                yield parts[1]
            elif fmt == "wiki":
                parts = line.split(" ")
                if len(parts) < 3:
                    continue
                try:
                    cnt = int(parts[2])
                except ValueError:
                    continue
                key = parts[0] + "/" + parts[1]
                # expand by count but cap to keep memory bounded; counts are the hotness signal
                for _ in range(min(cnt, 50)):
                    yield key
            elif fmt == "kv":
                yield line.split(",")[-1]
            else:
                raise ValueError(f"unknown fmt {fmt!r}")


def trace_to_sequence(paths, fmt: str, R: int = 256, window: int = 200_000,
                      max_windows: int = 400) -> np.ndarray:
    """Build a per-window segment-hotness matrix a_seq[t, i] from one or more trace files.

    Each window is `window` consecutive accesses; a_seq[t] = normalised count of accesses whose key
    hashes to each of R segments. Multiple files (e.g. consecutive hours) are concatenated in order,
    giving real temporal drift across windows."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rows = []
    counts = np.zeros(R, dtype=np.float64)
    seen = 0
    for p in paths:
        for key in iter_keys(p, fmt):
            counts[_seg(key, R)] += 1.0
            seen += 1
            if seen >= window:
                rows.append(counts / counts.sum())
                counts = np.zeros(R, dtype=np.float64)
                seen = 0
                if len(rows) >= max_windows:
                    return np.array(rows)
    if seen > window * 0.2:                          # keep a final partial window if substantial
        rows.append(counts / counts.sum())
    return np.array(rows)


def list_traces(data_dir, fmt: str):
    """Sorted trace files in a directory (so hourly/sequential files concatenate in time order)."""
    data_dir = Path(data_dir)
    exts = {"twitter": "*.csv*", "wiki": "*pageviews*", "kv": "*.txt*"}
    return sorted(data_dir.glob(exts.get(fmt, "*")))
