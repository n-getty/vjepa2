#!/usr/bin/env bash
# Build the DENSE pHash reference (eval yt_robotic_chole + surgenet_robotic).
# The first (sparse) build had 72% eval self-recall; this samples EVERY eval
# window clip + EVERY surgenet clip (4 frames each) via parallel workers, then
# finalizes. CPU-only frame decode.
#
# Fan-out: 9 eval-batch workers (1 Batch each) + N surgenet shard-range workers.
#
# Submit:
#   qsub scripts/build_phash_ref_pbs.sh
#
#PBS -N lemon_phash_ref
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=1:ncpus=104
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
umask 077

module load frameworks 2>/dev/null || module load frameworks

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
DATA=/flare/ModCon/ngetty/data
EVAL_DIR="$DATA/yt_chole_tool_windows"
SURG_DIR="$DATA/surg_vid_webdataset_resharded/surgenet_robotic"
OUT="$DATA/LEMON/phash_ref.json"
PDIR="$OUT.partials"
PYTHON="$(command -v python3)"
SURG_WORKERS="${SURG_WORKERS:-16}"

mkdir -p /flare/ModCon/ngetty/logs "$PDIR"
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"
"$PYTHON" -c "import av,cv2,numpy; print('av',av.__version__)"
rm -f "$PDIR"/*.json   # fresh build

pids=()
# eval: one worker per Batch dir (9 total)
NB=$("$PYTHON" -c "import glob;print(len(glob.glob('$EVAL_DIR/yt_robotic_chole_Batch*')))")
echo "eval batches: $NB"
for (( b=0; b<NB; b++ )); do
  "$PYTHON" "$ROOT/scripts/build_phash_ref.py" --pool eval \
      --eval-dir "$EVAL_DIR" --out "$OUT" --partial-dir "$PDIR" \
      --eval-frames-per-clip 4 --batch-start "$b" --batch-end $((b+1)) \
      > "$PDIR/_log_eval_$b.log" 2>&1 &
  pids+=($!)
done
# surgenet: fan shard ranges
NS=$("$PYTHON" -c "import glob;print(len(glob.glob('$SURG_DIR/*.tar')))")
echo "surgenet shards: $NS across $SURG_WORKERS workers"
CH=$(( (NS + SURG_WORKERS - 1) / SURG_WORKERS ))
for (( lo=0; lo<NS; lo+=CH )); do
  hi=$(( lo+CH )); (( hi>NS )) && hi=$NS
  "$PYTHON" "$ROOT/scripts/build_phash_ref.py" --pool surgenet \
      --surgenet-dir "$SURG_DIR" --out "$OUT" --partial-dir "$PDIR" \
      --train-frames-per-clip 4 --shard-start "$lo" --shard-end "$hi" \
      > "$PDIR/_log_surg_${lo}_${hi}.log" 2>&1 &
  pids+=($!)
done

fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "workers done, failed=$fail"; (( fail>0 )) && { echo "see $PDIR/_log_*.log" >&2; exit 1; }

"$PYTHON" "$ROOT/scripts/build_phash_ref.py" --finalize --out "$OUT" --partial-dir "$PDIR"
chmod 600 "$OUT"
echo "JOB END: $(date)"; ls -la "$OUT"
