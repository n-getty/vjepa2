#!/bin/bash
# 1-NODE GATE for the 256-node plan. Two things it must clear before any 16n
# arm (let alone prod time) is worth submitting:
#
#   leg bs1     -- per-rank batch_size=1 on the weight_distance_loss path.
#                  This is the documented "bs>=2" landmine (d_ij.unsqueeze(2)
#                  IndexError). masks_dist.py's squeeze(1) fix makes it legal and
#                  tests/models/test_masks_dist_batch1.py covers the tensor math,
#                  but it has NEVER been run end-to-end through the trainer.
#                  If it passes, 256n global batch is 3072 instead of 6144 --
#                  halving every large-batch recipe delta.
#
#   leg accum16 -- VJEPA_TRUE_ACCUM=16, the mechanism the 16n arms use to emulate
#                  the 256n batch. accum=2 is the ONLY value ever tried at scale
#                  and it HUNG at 16n (memory ga2-not-validated-16n). accum=16 is
#                  untested at any scale. A 1-node hang here costs an hour; the
#                  same hang at 16n costs 16 node-hours and a queue slot.
#
# Both legs run on the SMOKE config, same node, sequential. Iterations are scaled
# by 1/accum (40 at accum=1, 8 at accum=16) so the two legs do comparable WORK and
# both finish inside the per-leg timeout -- a fixed iter count would make the
# accum=16 leg ~46 min and get it killed as a false "hang".
#
# Submit directly (self-contained, ~1h):
#   qsub scripts/large_batch_smoke_1n.sh
# or run inside an existing held 1-node allocation:
#   bash scripts/large_batch_smoke_1n.sh
#
# NOTE: no `set -u` -- Aurora's lmod init references unbound vars and would abort
# the script at `module load` (memory set-u-module-load-trap).
#
#PBS -N lbgate
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

REPO=/lus/flare/projects/ModCon/ngetty/vjepa2
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
CFG=$REPO/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
OUTDIR=/flare/ModCon/ngetty/logs/lb_smoke_$(date +%y%m%d_%H%M%S)
mkdir -p "$OUTDIR"
cd "$REPO"

module load frameworks 2>/dev/null
export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate 2>/dev/null
export PYTHONPATH=$REPO:$PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MPICH_GPU_SUPPORT_ENABLED=1
# Single-node interactive: the pmix/mpi transport is fine here (the ofi/none
# combination is only required for multi-node HSDP subgroup collectives).
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
export FI_PROVIDER=cxi
export PYTHONFAULTHANDLER=1
export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export VJEPA_DIST_STRATEGY=hsdp
export LOCAL_WORLD_SIZE=12
export FSDP_SHARDING=shard_grad_op
export VJEPA_NUM_WORKERS=0
# Under qsub the script starts on the head compute node, but read the nodefile
# when present so MASTER_ADDR is the allocated host rather than wherever this
# shell happens to be.
if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
  export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  export MASTER_ADDR=$(hostname)
fi
export MASTER_PORT=29613
export WORLD_SIZE=12

# Per-leg config: batch_size AND folder, both baked into the YAML.
#
# `folder` must be written into the config, not passed as --folder: that flag is
# only honored in app/main_dist_aurora.py's submit() (the login-node path), while
# --train_mode reads params["folder"] straight from the YAML. Passing --folder to
# a --train_mode run is silently ignored, so both legs would share the SMOKE
# config's folder -- and since meta.load_checkpoint is true, leg 2 would RESUME
# leg 1's latest.pth.tar instead of starting clean, invalidating the gate.
# (The production launcher patches the YAML for the same reason.)
#
# load_checkpoint is also forced off: each leg must start from
# meta.pretrain_checkpoint so a stale latest.pth.tar in a reused folder cannot
# make a leg look healthy (or crash) for reasons unrelated to what we're testing.
make_cfg () {
  local bs=$1 out=$2 folder=$3 ipe=$4
  $PY - "$CFG" "$bs" "$out" "$folder" "$ipe" <<'PY'
import sys, yaml
src, bs, out, folder, ipe = (sys.argv[1], int(sys.argv[2]), sys.argv[3],
                             sys.argv[4], int(sys.argv[5]))
c = yaml.safe_load(open(src))
c["data"]["batch_size"] = bs
c["folder"] = folder
c["optimization"]["ipe"] = ipe
c.setdefault("meta", {})["load_checkpoint"] = False
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
print(f"wrote {out} (batch_size={bs}, ipe={ipe}, folder={folder}, "
      f"load_checkpoint=False)")
PY
}

run_leg () {
  local name=$1 bs=$2 accum=$3
  local folder=$OUTDIR/$name log=$OUTDIR/${name}.log cfg=$OUTDIR/${name}.yaml
  mkdir -p "$folder"
  # Scale ipe DOWN by accum. One accum=16 step does 16x the work of an accum=1
  # step (~68 s/iter measured vs ~6 s), so a fixed 40-iter leg would need ~46 min
  # and trip the 25-min timeout -- reporting a perfectly healthy run as a HANG.
  # The question this leg answers ("does high accum run at all, with finite
  # losses?") is settled in a handful of iterations; matching WORK per leg rather
  # than ITERS per leg keeps both legs inside the same wall-clock budget.
  local ipe=$(( 40 / accum )); (( ipe < 8 )) && ipe=8
  make_cfg "$bs" "$cfg" "$folder" "$ipe" || { echo "[$name] config gen FAILED"; return 1; }
  echo "===== LEG $name  (batch_size=$bs  VJEPA_TRUE_ACCUM=$accum) ====="
  local t0=$(date +%s)
  timeout 1500 mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 8 \
    env VJEPA_TRUE_ACCUM=$accum \
    python -m app.main_dist_aurora --train_mode \
      --fname "$cfg" --params_path "$cfg" \
      2>&1 | tee "$log"
  local rc=$? dt=$(( $(date +%s) - t0 ))
  if (( rc == 124 )); then   # timeout(1)
    echo "[$name] VERDICT: TIMEOUT after ${dt}s -- treat as a HANG, not a slow run."
    return 1
  fi
  # A leg passes only on real, FINITE losses -- not on rc=0, and not on row count
  # alone. Column 3 of log_r0.csv is `loss` (see the CSVLogger spec in
  # app/vjepa_2_1/train.py). Grepping the log text for "nan" is useless here: it
  # matches unrelated words and says nothing about the actual values, so the loss
  # column is parsed directly.
  local csv="$folder/log_r0.csv" rows=0 bad=0 last=""
  if [[ -f "$csv" ]]; then
    rows=$(awk -F, '$2 ~ /^[0-9]+$/ {n++} END{print n+0}' "$csv")
    bad=$(awk -F, '$2 ~ /^[0-9]+$/ {
            v=$3+0
            if ($3 ~ /[Nn][Aa][Nn]|[Ii][Nn][Ff]/ || v != v || v == 0) n++
          } END{print n+0}' "$csv")
    last=$(awk -F, '$2 ~ /^[0-9]+$/ {v=$3} END{print v}' "$csv")
  fi
  echo "[$name] rc=$rc  ${dt}s  csv_rows=$rows  nonfinite_or_zero_loss=$bad  last_loss=${last:-n/a}"
  if (( rc != 0 )); then echo "[$name] VERDICT: FAIL (rc=$rc)"; return 1; fi
  if (( rows < 5 )); then echo "[$name] VERDICT: FAIL (only $rows iters logged)"; return 1; fi
  if (( bad > 0 )); then
    echo "[$name] VERDICT: FAIL ($bad/$rows rows have NaN/Inf/zero loss)"; return 1
  fi
  echo "[$name] VERDICT: PASS ($rows iters, final loss $last)"
  return 0
}

FAILED=0
run_leg bs1     1 1  || FAILED=1
run_leg accum16 2 16 || FAILED=1

# Verdict also goes to a DETERMINISTIC path. PBS names its own log
# <jobid>.<server>.OU, which nothing downstream can predict, so a watcher has to
# guess the filename. This file is always here.
VERDICT_FILE=/flare/ModCon/ngetty/logs/lbgate_VERDICT.txt
{
  echo
  echo "================ SUMMARY ================"
  echo "job:  ${PBS_JOBID:-interactive}   $(date)"
  echo "logs: $OUTDIR"
  if (( FAILED )); then
    echo "GATE FAILED -- do NOT submit the 16n arms yet."
    echo "  bs1 fail     -> keep per-rank bs=2; 256n global batch is 6144, not 3072."
    echo "  accum16 fail -> the arms cannot emulate the 256n batch this way;"
    echo "                  bisect accum (2,4,8) before spending 16n time."
  else
    echo "GATE PASSED -- ./scripts/submit_large_batch_arm.sh lbA (and lbB)"
  fi
} | tee "$VERDICT_FILE"
(( FAILED )) && exit 1
exit 0
