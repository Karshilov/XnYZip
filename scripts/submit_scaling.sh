#!/bin/bash -l
# Submit one strong-scaling sweep on Bebop: one PBS job per node count, each
# job loops over block sizes internally (see run_scaling.pbs).
#
# Required env vars:
#   INPUT     absolute path to the input file
#   BASE      label used for output dir, e.g. "hacc"
# Optional:
#   NODES         space-sep node counts        (default "1 2 4 8 16")
#   PPN           processes per node           (default 36 for Bebop Broadwell)
#   MEM           per-node memory request      (default 128gb)
#   BLOCKS_MIB    block sizes (forwarded)      (default "64 128 256")
#   BOUND         absolute L2 bound            (REQUIRED; pre-multiply diag)
#   MODE          -normal | -rle               (default -rle)
#   QUANTIZER, CURVE, DIRECT_THRESH (forwarded as-is)
#   PBS_QUEUE     -q value                     (default unset → cluster default)
#   RUN_ROOT      where outputs go             (default $PWD/runs)
#   XNYZIP_MPI    binary path                  (default $PWD/build/XnYZip_mpi)

set -euo pipefail

: "${INPUT:?INPUT must be set}"
: "${BASE:?BASE must be set}"
: "${BOUND:?BOUND (absolute L2) must be set; e.g. for HACC at eb=1e-3, BOUND = 1e-3 * diag}"
: "${NODES:=1 2 4 8 16}"
: "${PPN:=36}"
: "${MEM:=128gb}"
: "${BLOCKS_MIB:=64 128 256}"
: "${MODE:=-rle}"
: "${QUANTIZER:=to}"
: "${CURVE:=-z}"
: "${DIRECT_THRESH:=0}"  # block mode; 1024 default has direct-mode bug
: "${RUN_ROOT:=$PWD/runs}"
: "${XNYZIP_MPI:=$PWD/build/XnYZip_mpi}"

EXTRA_QSUB=()
[[ -n "${PBS_QUEUE:-}" ]] && EXTRA_QSUB+=(-q "$PBS_QUEUE")

PBS_SCRIPT="$(cd "$(dirname "$0")" && pwd)/run_scaling.pbs"
mkdir -p "$RUN_ROOT"

for N in $NODES; do
    OUTROOT="$RUN_ROOT/${BASE}_n${N}"
    mkdir -p "$OUTROOT"
    SELECT="select=${N}:ncpus=${PPN}:mpiprocs=${PPN}:mem=${MEM}"
    echo "==> submitting N=$N  ($((N * PPN)) ranks)  -> $OUTROOT"
    qsub \
        "${EXTRA_QSUB[@]}" \
        -l "$SELECT" \
        -v "INPUT=$INPUT,OUTROOT=$OUTROOT,BLOCKS_MIB=$BLOCKS_MIB,BOUND=$BOUND,QUANTIZER=$QUANTIZER,CURVE=$CURVE,MODE=$MODE,DIRECT_THRESH=$DIRECT_THRESH,XNYZIP_MPI=$XNYZIP_MPI" \
        "$PBS_SCRIPT"
done

echo
echo "After all jobs finish, aggregate with:"
echo "  python scripts/parse_scaling_results.py $RUN_ROOT --base $BASE \\"
echo "         --output scaling_${BASE}.xlsx"
