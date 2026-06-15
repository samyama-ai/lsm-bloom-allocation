#!/usr/bin/env bash
# H3 (real engine): validate the core model primitive  FPR = exp(-ln^2(2) * bits_per_key)  on a
# real RocksDB, and the Monkey direction (optimize_filters_for_hits drops the deepest-level filter
# => big memory saving for little FPR cost). Uses db_bench from rocksdb-tools. No mocks.
#
#   bench/rocksdb_fpr.sh <out_json> [num_keys] [reads]
set -euo pipefail
OUT=${1:-results/rocksdb_fpr.json}; NUM=${2:-3000000}; READS=${3:-2000000}
DB=/tmp/rdb_fpr
mkdir -p "$(dirname "$OUT")"
echo "[" > "$OUT"; first=1

emit() { # bits useful fullpos ofh
  fpr=$(python3 -c "u=$2;p=$3;print(p/(u+p) if (u+p)>0 else float('nan'))")
  pred=$(python3 -c "import math;print(math.exp(-(math.log(2)**2)*$1))")
  [ $first -eq 0 ] && echo "," >> "$OUT"; first=0
  printf '  {"bits":%s,"useful":%s,"full_positive":%s,"fpr_emp":%s,"fpr_pred":%s,"ofh":%s}' \
    "$1" "$2" "$3" "$fpr" "$pred" "$4" >> "$OUT"
}

run() { # bits optimize_filters_for_hits
  local bits=$1 ofh=$2
  rm -rf "$DB"
  out=$(db_bench --db="$DB" --benchmarks=fillrandom,readmissing --num="$NUM" --reads="$READS" \
        --bloom_bits="$bits" --optimize_filters_for_hits="$ofh" --compression_type=none \
        --cache_size=8388608 --statistics --disable_wal=1 2>&1 || true)
  useful=$(echo "$out"  | grep -oE 'filter.useful COUNT : [0-9]+'        | grep -oE '[0-9]+$' | tail -1)
  fullpos=$(echo "$out" | grep -oE 'filter.full.positive COUNT : [0-9]+' | grep -oE '[0-9]+$' | tail -1)
  useful=${useful:-0}; fullpos=${fullpos:-0}
  echo "bits=$bits ofh=$ofh useful=$useful full_positive=$fullpos"
  emit "$bits" "$useful" "$fullpos" "$ofh"
}

for b in 2 4 6 8 10 12 14 16; do run "$b" 0; done   # FPR curve
for b in 8 10; do run "$b" 1; done                   # Monkey-direction (drop deepest filter)
echo "" >> "$OUT"; echo "]" >> "$OUT"
echo "wrote $OUT"
