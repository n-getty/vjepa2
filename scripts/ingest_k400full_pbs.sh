#!/bin/bash -l
# Ingest raw full K400 (part_*.tar.gz of clipped mp4s) -> project WebDataset triples.
# I/O-bound repack (no re-encode); fans workers over parts via mpiexec, then merges manifests.
#
# Submit (2 nodes, debug queue is plenty — this is minutes not hours):
#   qsub -A AuroraGPT -q debug -l select=2 -l walltime=00:30:00 \
#        -l filesystems=home:flare scripts/ingest_k400full_pbs.sh
#
#PBS -N k400_ingest
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/k400_ingest.log

set -o pipefail   # NOT set -u (lmod ZSH_EVAL_CONTEXT trap)

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
INPUT=/flare/ModCon/ngetty/data/kinetics400_full/train
OUTPUT=/flare/ModCon/ngetty/data/kinetics400_full_wds/kinetics400
SHARDS_PER_WORKER=${SHARDS_PER_WORKER:-19}   # 48 workers * 19 = 912 shards (~267 clips/shard @243K)
PPN=${PPN:-24}                                # workers per node (I/O-bound, oversubscribe cores)

cd "$ROOT"
mkdir -p /flare/ModCon/ngetty/logs "$OUTPUT"
module load frameworks

NODES=$(wc -l < "$PBS_NODEFILE")
NUM_WORKERS=$(( NODES * PPN ))
echo "K400 ingest: $NODES nodes x $PPN = $NUM_WORKERS workers, $SHARDS_PER_WORKER shards/worker"
echo "input=$INPUT  output=$OUTPUT"

# One MPI world across all nodes; each rank picks up its worker-id from PMI_RANK.
mpiexec --pmi=pmix -n "$NUM_WORKERS" -ppn "$PPN" --cpu-bind depth --depth 4 \
  bash -c 'python3 '"$ROOT"'/scripts/ingest_k400full_to_wds.py \
      --input '"$INPUT"' --output '"$OUTPUT"' \
      --shards-per-worker '"$SHARDS_PER_WORKER"' \
      --num-workers '"$NUM_WORKERS"' --worker-id ${PMI_RANK:-${PALS_RANKID:-0}}'
rc=$?
echo "all workers rc=$rc"

if [ $rc -eq 0 ]; then
  echo "=== merging manifests ==="
  python3 "$ROOT"/scripts/ingest_k400full_to_wds.py \
      --output "$OUTPUT" --merge --num-workers "$NUM_WORKERS"
  echo "=== final metadata.json ==="
  head -5 "$OUTPUT/metadata.json"
fi
exit $rc
