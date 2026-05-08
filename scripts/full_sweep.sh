#!/bin/bash
# Comprehensive chunk-size sweep across all datasets and error bounds.
# Outputs one xlsx per (dataset, eb_scale).
#
# Runs in background; prints progress to log.

set -uo pipefail
cd "$(dirname "$0")/.."

EXE=./build/XnYZip
[[ -x "$EXE" ]] || { echo "build XnYZip first"; exit 1; }

mkdir -p sweep_results
LOG=sweep_results/full_sweep.log
echo "starting full sweep at $(date)" > "$LOG"

# (dataset_path, label, chunk_sizes_mib)
DATASETS=(
    "/home/youyuan/data-preparation/dragon_vrip.bin|dragon|0.5 1 2 4"
    "/home/youyuan/data-preparation/md/vesicles_1_75M.f32|vesicles|1 2 4 8 16"
    "/home/youyuan/data-preparation/usgs/usgs.bin|usgs|4 16 64 128"
    "/home/youyuan/data-preparation/hacc_256M.bin|hacc256M|4 16 64 128"
)
EB_SCALES="1e-3 1e-4 1e-5"
VARIANTS="to-z-normal"

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r path label sizes <<< "$entry"
    for eb in $EB_SCALES; do
        OUT=sweep_results/sweep_${label}_eb${eb}.xlsx
        TMP=sweep_results/tmp_${label}_eb${eb}
        mkdir -p "$TMP"
        echo "=== $(date -Iseconds) $label eb=$eb ===" >> "$LOG"
        python3 chunk_sweep.py \
            --input "$path" \
            --bound-scale "$eb" \
            --chunk-sizes-mib $sizes \
            --variants $VARIANTS \
            --output "$OUT" \
            --temp-dir "$TMP" \
            >> "$LOG" 2>&1
        rm -rf "$TMP"
    done
done

echo "done at $(date)" >> "$LOG"
