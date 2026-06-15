#!/usr/bin/env bash
# Fetch REAL workload traces for the H2c real-trace-transfer test. No data is vendored.
#
# Twitter production cache traces (Yang et al., OSDI'20), hosted at CMU PDL. Files clusterN.sort.zst
# are time-sorted; we stream-decompress a PREFIX (earliest time period) of each cluster so we don't
# download the full 20GB+ archive. Each prefix is a real, contiguous slice of production traffic.
#
#   bench/fetch_data.sh <out_dir> [max_lines] [cluster_ids...]
# e.g. bench/fetch_data.sh ./data 100000000 45 34 22
set -euo pipefail
BASE=https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source
OUT=${1:-./data}; MAX=${2:-100000000}; shift 2 || true
CLUSTERS=("$@"); [ ${#CLUSTERS[@]} -eq 0 ] && CLUSTERS=(45 34 22)
mkdir -p "$OUT"
for n in "${CLUSTERS[@]}"; do
  dst="$OUT/cluster${n}.csv"
  if [ -s "$dst" ]; then echo "have $dst"; continue; fi
  echo "streaming cluster$n prefix ($MAX lines) -> $dst"
  # SIGPIPE from head stops curl early, so only the needed prefix is downloaded.
  curl -s "$BASE/cluster${n}.sort.zst" | zstd -dc 2>/dev/null | head -n "$MAX" > "$dst" || true
  echo "  $(wc -l < "$dst") lines, $(du -h "$dst" | cut -f1)"
done
echo "DONE"
