#!/usr/bin/env bash
# Build the CROP-AUGMENTED eval reference for eval-leak scrubbing. Hashes every
# eval window clip WITH crop (0.9/0.8/0.7) + circular-mask variants so LEMON's
# cropped/masked copies of eval videos are (best-effort) caught -- raw frame
# pHash is crop-blind. Fan 9 workers (one per Batch), then finalize.
#
# Submit:  qsub scripts/build_eval_aug_ref_pbs.sh
#
#PBS -N eval_aug_ref
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
REF="$DATA/LEMON/eval_aug_ref.json"
PYTHON="$(command -v python3)"

mkdir -p /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-}"
rm -f "$REF.partials"/*.json 2>/dev/null || true

NB=$("$PYTHON" -c "import glob;print(len(glob.glob('$EVAL_DIR/yt_robotic_chole_Batch*')))")
echo "eval batches: $NB"
pids=()
for (( b=0; b<NB; b++ )); do
  "$PYTHON" "$ROOT/scripts/scrub_eval_leakage.py" --mode build-eval-ref \
      --eval-dir "$EVAL_DIR" --eval-ref "$REF" \
      --eval-frames-per-clip 4 --batch-start "$b" --batch-end $((b+1)) \
      > "$REF.partials/_log_$b.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
(( fail>0 )) && { echo "workers failed=$fail" >&2; tail -5 "$REF.partials"/_log_*.log; exit 1; }

"$PYTHON" "$ROOT/scripts/scrub_eval_leakage.py" --mode finalize-eval-ref --eval-ref "$REF"
chmod 600 "$REF"
echo "JOB END: $(date)"; ls -la "$REF"
