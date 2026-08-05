#!/bin/bash
# 256-node training reading from DAOS instead of node-local /tmp staging.
#
# WHAT CHANGED vs scripts/vitG384_256n_shakeout.sh
# ------------------------------------------------
# No staging step at all. The corpus lives once in AuroraGPT/vjepa_surg_wds and
# every node reads it over the fabric. Staging was copying 265 GB/node = 66 TB
# aggregate to deliver ~58 GB/node of actual reads (4.6x write amplification),
# at a measured 0.43 GB/s/node -- up to ~2.7 h before a single iteration.
# Streaming demand during the run is only ~1.5-3 GB/s aggregate, because 15 TB
# of reads spread over hours instead of minutes.
#
# THREE THINGS THAT MUST BE RIGHT (each fails silently or hangs):
#
#   WDS_LOCAL_SLICING=0. This is the subtle one. The flag exists only because
#   each node's /tmp held a DIFFERENT shard subset, so slicing by local
#   rank/world (12) was the correct way to cover it. On DAOS every node sees the
#   WHOLE corpus, so local slicing makes all 256 nodes compute the SAME 12
#   slices -- rank 0 on every node reads identical shards. That is 256x data
#   duplication with no error message. Global slicing (=0) gives each of the
#   3072 ranks a disjoint slice.
#
#   mpiexec --no-vni. Without it DAOS RPCs over libfabric fail with
#   NA_HOSTUNREACH and the job hangs.
#
#   libpil4dfs stays OFF. It hangs FSDP AllGather (DAOS-17499) and we run HSDP.
#   PRISM also measured it as no benefit -- data load is <0.01 s/step anyway.
#
# -l filesystems must include daos_user_fs (the agent only starts in the PBS
# prologue) -- but never add it to non-DAOS jobs: they queue forever when DAOS
# is down.
#
#   qsub scripts/vitG384_256n_daos.sh
#
# NOTE: no `set -u` (memory set-u-module-load-trap).
#
#PBS -N vg256d
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=256
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
POOL=${DAOS_POOL:-AuroraGPT}
CONT=${DAOS_CONT:-vjepa_surg_wds}
DAOS_MNT=/tmp/${POOL}/${CONT}
# Model weights live in their OWN container. Job 8730487 died learning why:
# the corpus was on DAOS but `pretrain_checkpoint` was still a 28.2 GB file on
# Lustre, and all 3072 ranks read it independently -- up to 85 TB against a
# filesystem measured at 2.71 GB/s. Ranks cleared the checkpoint load at only
# ~69/min, so the job would have spent its whole hour loading and never reached
# an iteration. Invisible at 16n (192 ranks x 28 GB is tolerable); fatal at 3072.
#
# Note the oclass differs from the corpus container ON PURPOSE: vjepa_models uses
# EC_16P3GX (stripe each file across ALL servers -- right for ONE big file every
# rank reads), while vjepa_surg_wds uses EC_16P3G32 (right for thousands of
# independent shards). Same reasoning, opposite answer.
MODELS_CONT=${DAOS_MODELS_CONT:-vjepa_models}
MODELS_MNT=/tmp/${POOL}/${MODELS_CONT}
PPN=12
NNODES=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l); NNODES=${NNODES:-256}
WORLD=$(( NNODES * PPN ))
SHAKE_IPE=${VJEPA_SHAKE_IPE:-50}
# SUSTAINED mode (VJEPA_SUSTAINED=1): keep the config's real epoch count, save
# every epoch, and self-resubmit on exit. Default 0 = shakeout, unchanged.
#
# Why the shakeout defaults are wrong for a real run: it forces epochs=1, so with
# save_every_freq=5 no checkpoint is ever written and a watchdog kill loses
# everything. Sustained training needs each epoch DURABLE, and the epoch has to
# be shorter than the mean time to failure or the run never banks progress. At
# 256n that estimate is ~14 min (16n's documented ~3.5-4 h scaled by node count),
# and at ~10 s/iter ipe=30 is ~5 min/epoch -- the same reasoning behind the 16n
# recipe's "ipe=30 bounds loss to <1 epoch", with a 16x tighter deadline.
SUSTAINED=${VJEPA_SUSTAINED:-0}
SUSTAINED_IPE=${VJEPA_SUSTAINED_IPE:-30}

# vitG384_lbA is `batch_size: 2` -- the max-throughput lever -- with lr/ema/
# warmup/lambda all DERIVED for the gb=6144 that 3072 ranks x 2 produces. At this
# node count it is a NATIVE gb=6144 config, not an emulation of one: lbA and lbA8
# are byte-identical in meta/data/mask/loss/data_aug apart from batch_size, and
# see the same 3.83 M samples (624 x 6144 == 1248 x 3072), so the replay position
# is unchanged. lbA8 (bs=1, gb=3072) remains the accum fallback -- see
# VJEPA_TRUE_ACCUM below.
CFG_NAME=${VJEPA_CFG_NAME:-vitG384_lbA}
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
# PER-JOB output dir. This is a shakeout, not a resumable production run, so
# every job gets its own directory keyed by jobid and node count.
#
# Sharing one dir across runs is what made log_r0.csv a cross-job append target,
# and that single fact produced three separate false readings today: a verdict
# crediting this job with the previous run's 4 iterations, an mtime-based guard
# that could not fix it (appending refreshes mtime), and a monitor reporting a
# queued 64n job as "11 iters" from the finished 16n run's tail. Isolating the
# directory removes the root cause instead of teaching every consumer to parse
# around it.
#
# Set VJEPA_CKPT_DIR explicitly for a run that is meant to resume.
_TAG="${PBS_JOBID%%.*}"; _TAG="${_TAG:-manual}"
CKPT_DIR=${VJEPA_CKPT_DIR:-/flare/ModCon/ngetty/checkpoints/daos_shakeout/${CFG_NAME}_n${NNODES}_${_TAG}}
PARAMS=$CKPT_DIR/params-pretrain.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
# Keyed by node count: this launcher is run at 2n (validation) and 64n as well as
# 256n, and a single fixed path means the small run silently overwrites the big
# run's record. Same reasoning as the per-job CKPT_DIR above.
VERDICT=${VJEPA_VERDICT:-/flare/ModCon/ngetty/logs/daos_n${NNODES}_VERDICT.txt}
mkdir -p "$CKPT_DIR" /flare/ModCon/ngetty/logs

echo "JOB START $(date) PBS_JOBID=$PBS_JOBID"
echo "topology ${NNODES}n x ${PPN} = ${WORLD} ranks; config=$CFG_NAME; data=DAOS $POOL:$CONT"
[[ -f "$BASE_CFG" ]] || { echo "FATAL: missing $BASE_CFG"; exit 1; }

cd $ROOT
module use /soft/modulefiles
module load frameworks
module load daos
export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
# The measured recipe -- CCL transport, FI, HSDP, WDS_LOCAL_SLICING=0, the
# LD_PRELOAD and CCL_KVS_MODE unsets. This script was the only place it lived;
# it is now shared so a recipe change lands everywhere at once. Overrides go
# BELOW the source (most values in the fragment are ${VAR:-default}; the FI ones
# are hard-set because Aurora's profile already exports them).
source $ROOT/scripts/lib/aurora_hsdp_env.sh
export LOCAL_WORLD_SIZE=$PPN
# accum stays at 1. The lever is 2 clips per collective, and `batch_size: 2` in
# the config supplies them -- at MATCHED global batch (job 8735877, 64n, n=19)
# bs=2 ran 24.92 s / 61.6 clips/s with 4.6 GiB min l0-free against accum's
# 32.36 s / 47.5 clips/s / 0.8 GiB. Medians 1.30x with overlapping IQRs, so
# throughput is suggestive and headroom is decisive; nothing favours accum.
#
# The older "+36%" number (job 8731439) compared accum=2 against HALF the global
# batch. It showed "2 clips per collective beats 1", not "accum beats bs=2".
#
# accum IS the fallback if 256n OOMs at bs=2, but it has to keep gb=6144 to keep
# lbA's schedule valid, so it is bs=1 AND accum=2 TOGETHER -- on lbA, not lbA8:
#
#     VJEPA_PER_RANK_BS=1 VJEPA_TRUE_ACCUM=2 qsub scripts/vitG384_256n_daos.sh
#
# Switching to lbA8 instead would take lbA8's gb=3072 constants at gb=6144; the
# assertion in the heredoc rejects that. Raising accum alone does too (gb 12288).
export VJEPA_TRUE_ACCUM=${VJEPA_TRUE_ACCUM:-1}

if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR
export MASTER_PORT=29500
export WORLD_SIZE=$WORLD
echo "MASTER_ADDR=$MASTER_ADDR WORLD_SIZE=$WORLD_SIZE WDS_LOCAL_SLICING=$WDS_LOCAL_SLICING"

# Mount the container on every node.
timeout 600 launch-dfuse.sh ${POOL}:${MODELS_CONT} || { echo "FATAL: launch-dfuse (models) failed"; exit 1; }
timeout 60 ls "$MODELS_MNT" >/dev/null 2>&1 || { echo "FATAL: $MODELS_MNT unresponsive"; exit 1; }
echo "models container mounted: $(ls "$MODELS_MNT" 2>/dev/null | tr '\n' ' ')"
timeout 600 launch-dfuse.sh ${POOL}:${CONT} || { echo "FATAL: launch-dfuse failed"; exit 1; }
mount | grep -q "$CONT" || { echo "FATAL: not mounted at $DAOS_MNT"; exit 1; }
# A hung dfuse presents as a hang much later, in the loader, on one rank. Catch
# it here where the message is unambiguous.
timeout 60 ls "$DAOS_MNT" >/dev/null 2>&1 || { echo "FATAL: $DAOS_MNT unresponsive"; exit 1; }
echo "DAOS mounted; sources: $(ls "$DAOS_MNT" 2>/dev/null | tr '\n' ' ')"

$PY $ROOT/scripts/prepare_runtime_config.py \
  $BASE_CFG --root $ROOT --num-gpus $PPN --num-nodes $NNODES --weak-scale > /dev/null || {
    echo "FATAL: prepare_runtime_config failed"; exit 1; }
RUNTIME_CFG=$ROOT/.runtime_configs/n${NNODES}g${PPN}_weak/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
cp "$RUNTIME_CFG" "$PARAMS" || { echo "FATAL: no runtime config"; exit 1; }
$PY - "$PARAMS" "$CKPT_DIR" "$SHAKE_IPE" "$MODELS_MNT" "$WORLD" <<'PY'
import sys, yaml, os
p, d, ipe, models = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
world = int(sys.argv[5])
c = yaml.safe_load(open(p))
c["folder"] = d
opt = c["optimization"]
orig_ipe, orig_epochs = opt["ipe"], opt["epochs"]

# ---- GLOBAL-BATCH ASSERTION -------------------------------------------------
# lr / ema / warmup / lambda in these configs are DERIVED for one specific global
# batch (scripts/gen_large_batch_configs.py). Run the config at a different one
# and every constant is silently wrong -- no error, just a worse model, which is
# the failure mode of [[step-count-constants-break-at-scale]].
#
# The whole family is derived to hold SAMPLES SEEN invariant at the 16n baseline
# 30 x 333 x 384, so gb_derived = SAMPLES / (ipe x epochs) recovers the batch a
# config was built for without needing it recorded anywhere. Tolerance covers the
# epoch rounding in derive() (624 x 6144 is 0.06% off the target); it is nowhere
# near loose enough to admit a 2x error.
#
# What this catches: scripts/submit_large_batch_arm.sh pairs lbA with ACCUM=16,
# which is correct at 16 nodes (where accum EMULATES gb=6144) and gives gb=98304
# if the same pairing is ever pointed at 256.
SAMPLES = 30 * 333 * 384  # 3,836,160
accum = int(os.environ.get("VJEPA_TRUE_ACCUM", "1"))
# VJEPA_PER_RANK_BS exists so the OOM fallback is reachable WITHOUT changing the
# schedule: bs=1 + accum=2 on lbA is the same gb=6144 and the same lr/ema/warmup
# as bs=2 + accum=1, just paying accum's per-step overhead to get flat
# activations. Doing it by switching to lbA8 instead would silently take lbA8's
# gb=3072 constants -- the assertion below rejects that, correctly.
_bs_override = os.environ.get("VJEPA_PER_RANK_BS")
if _bs_override:
    print(f"batch_size {c['data']['batch_size']} -> {_bs_override} "
          f"(VJEPA_PER_RANK_BS)")
    c["data"]["batch_size"] = int(_bs_override)
bs = c["data"]["batch_size"]
gb_actual = world * bs * accum
gb_derived = SAMPLES / (orig_ipe * orig_epochs)
print(f"global batch: {world} ranks x bs {bs} x accum {accum} = {gb_actual}; "
      f"config schedule derived for ~{gb_derived:.0f} "
      f"({orig_ipe} x {orig_epochs} steps)")
if abs(gb_actual - gb_derived) / gb_derived > 0.05:
    msg = (f"global batch {gb_actual} does not match the {gb_derived:.0f} this "
           f"config's lr/ema/warmup/lambda were derived for. Either pick the "
           f"config for this topology (at 3072 ranks: vitG384_lbA = bs2/gb6144, "
           f"vitG384_lbA8 = bs1/gb3072) or set VJEPA_TRUE_ACCUM to match.")
    if os.environ.get("VJEPA_SKIP_GB_CHECK", "0") == "1":
        print(f"WARNING: {msg}\n  OVERRIDDEN by VJEPA_SKIP_GB_CHECK=1 -- the "
              f"schedule constants are wrong for this batch.")
    else:
        sys.exit(f"FATAL: {msg} Set VJEPA_SKIP_GB_CHECK=1 only if being "
                 f"off-schedule is deliberate.")

sustained = os.environ.get("VJEPA_SUSTAINED", "0") == "1"
if sustained:
    # Shorten ipe so each epoch banks fast and make every epoch durable -- but
    # PRESERVE TOTAL STEPS. The previous version kept `epochs` while overwriting
    # `ipe`, which turned lbA's derived 39 x 16 = 624 into 30 x 16 = 480: 2.95 M
    # samples instead of 3.83 M, run against an EMA/warmup/lambda schedule built
    # for 624. Same class of bug as the gb assertion above -- a step count
    # changing under constants that were derived from it.
    new_ipe = int(os.environ.get("VJEPA_SUSTAINED_IPE", "30"))
    opt["ipe"] = new_ipe
    opt["epochs"] = max(1, round(orig_ipe * orig_epochs / new_ipe))
    # warmup is specified in EPOCHS but consumed as int(warmup * ipe)
    # (app/vjepa_2_1/utils.py:493), so shortening the epoch shortens warmup in
    # steps unless it is rescaled. Hold the STEP count.
    opt["warmup"] = round(opt["warmup"] * orig_ipe / new_ipe, 3)
    c.setdefault("meta", {})["save_every_freq"] = 1
    print(f"SUSTAINED: ipe {orig_ipe}->{new_ipe} epochs {orig_epochs}->"
          f"{opt['epochs']} (steps {orig_ipe*orig_epochs}->"
          f"{new_ipe*opt['epochs']}) warmup {opt['warmup']} ep = "
          f"{int(opt['warmup']*new_ipe)} steps save_every_freq=1")
else:
    opt["ipe"] = ipe
    opt["epochs"] = 1
# Repoint the init checkpoint at DAOS. Leaving it on Lustre is what killed job
# 8730487: 3072 ranks each pulling 28.2 GB off a 2.71 GB/s filesystem.
meta = c.setdefault("meta", {})
ck = meta.get("pretrain_checkpoint")
if ck:
    cand = os.path.join(models, os.path.basename(ck))
    if os.path.exists(cand):
        meta["pretrain_checkpoint"] = cand
        print(f"pretrain_checkpoint -> {cand} (DAOS)")
    else:
        # Do not silently fall back to Lustre: that is precisely the failure we
        # are fixing, and at 3072 ranks it burns the whole allocation.
        sys.exit(f"FATAL: {cand} missing -- stage the checkpoint into DAOS first")
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
# Report what was WRITTEN, not what the shakeout branch would have written --
# the old line hardcoded "epochs=1" and so lied in sustained mode.
print(f"folder={d} ipe={opt['ipe']} epochs={opt['epochs']} "
      f"bs={bs} accum={accum} lr={opt['lr']} ema={opt['ema']}")
PY

# --local_data_root remaps every dataset dir by BASENAME
# (app/main_dist_aurora.py:366-370), so pointing it at the DAOS mount is a
# drop-in -- no config edit, no trainer change.
# ---- STALL WATCHDOG + FORENSICS (ported from the proven 16n launcher).
#
# This is the difference between a shakeout and a run that survives. Documented
# at 16 nodes: ~1 disruptive event per 3.5-4 h, a cohort-wide FSDP collective
# desync, ALL of them self-healed -- that is what carried e15->e215 autonomously.
# If per-node hazard is constant, MTTF scales down with node count:
#
#     16n  ~225 min      64n  ~56 min      256n  ~14 min
#
# So at 3072 ranks a hang is expected roughly EVERY 14 MINUTES, and a one-hour
# slot should see several. Without this layer the FIRST one ends the run and the
# block is spent; with it, the run keeps going.
#
# The deadlines are scaled from the 16n values, not copied: startup at 256n is
# rendezvous + a 28 GB DAOS checkpoint load across 3072 ranks, measured at ~7 min
# for 24 ranks, so FIRST_ITER gets 30 min. STALL stays at 1800 s -- comfortably
# above the worst recoverable spike observed (~544 s) so it fires only on a true
# hang, not on the intrinsic fabric tax.
CSV_WATCH="$CKPT_DIR/log_r0.csv"
STALL_DEADLINE=${VJEPA_STALL_DEADLINE:-1800}
FIRST_ITER_DEADLINE=${VJEPA_FIRST_ITER_DEADLINE:-1800}
DIAG_DIR="$CKPT_DIR/hang_diag"; mkdir -p "$DIAG_DIR"

capture_hang_forensics() {
    local tag="$1" stamp; stamp=$(date +%Y%m%d_%H%M%S)
    echo "WATCHDOG: capturing forensics ($tag) -> $DIAG_DIR/hang_${stamp}_*" >&2
    # Node attribution across incidents: which physical hosts keep appearing?
    [ -f "${PBS_NODEFILE:-}" ] && cp "$PBS_NODEFILE" "$DIAG_DIR/hang_${stamp}_nodefile.txt" 2>/dev/null
    echo "jobid=$PBS_JOBID nodes=$NNODES last_csv_row=$(tail -1 "$CSV_WATCH" 2>/dev/null)" \
        > "$DIAG_DIR/hang_${stamp}_info.txt"
    # SIGUSR1 -> the armed faulthandler dumps every rank's all-thread stack and
    # CONTINUES, so we learn which collective each rank is blocked in before the
    # hard kill. Best-effort; never let diagnostics block the kill.
    mpiexec -n $NNODES -ppn 1 --cpu-bind none --no-vni \
        bash -c 'pkill -USR1 -f app.main_dist_aurora 2>/dev/null; true' >/dev/null 2>&1 || true
    sleep 20
}

(
    start=$(date +%s); last_rows=-1; last_change=$start
    while true; do
        sleep 60
        pgrep -f "app.main_dist_aurora" >/dev/null 2>&1 || exit 0
        now=$(date +%s)
        rows=0; [ -f "$CSV_WATCH" ] && rows=$(awk -F, '$2 ~ /^[0-9]+$/ {n++} END{print n+0}' "$CSV_WATCH" 2>/dev/null)
        if [ "${rows:-0}" -gt 0 ]; then
            if [ "$rows" -ne "$last_rows" ]; then last_rows=$rows; last_change=$now; fi
            if [ $((now-last_change)) -gt $STALL_DEADLINE ]; then
                echo "WATCHDOG: STALL -- no new iters for ${STALL_DEADLINE}s at row $rows." >&2
                capture_hang_forensics "stall_row${rows}"
                pkill -9 -f "app.main_dist_aurora"; exit 1
            fi
        elif [ $((now-start)) -gt $FIRST_ITER_DEADLINE ]; then
            echo "WATCHDOG: NO FIRST ITER within ${FIRST_ITER_DEADLINE}s." >&2
            capture_hang_forensics "no_first_iter"
            pkill -9 -f "app.main_dist_aurora"; exit 1
        fi
    done
) &
WATCHDOG_PID=$!
export VJEPA_ITER_WATCHDOG_S=${VJEPA_ITER_WATCHDOG_S:-600}

T0=$(date +%s)
# PER-RANK OUTPUT, not one funnel. Job 8730678 pushed 442,649 stdout lines from
# 3072 ranks through the head node's PBS log while that same node served the
# rendezvous store and answered DAOS agent keepalives -- and it was the head node
# whose DAOS agent then missed a 120 s ping and killed the run. `--outfile-pattern`
# writes each rank's stdout ON THE NODE WHERE THAT RANK RUNS (per the mpiexec man
# page), so the funnel disappears rather than merely shrinking.
#
# Rank 0's file is the one to read; the rest exist for post-mortems. The PBS log
# keeps the launcher's own output (mount checks, verdict), which is what makes
# the job diagnosable at a glance.
RANKLOG=${VJEPA_RANKLOG_DIR:-/flare/ModCon/ngetty/logs/ranklogs_${PBS_JOBID%%.*}}
mkdir -p "$RANKLOG"
echo "per-rank stdout -> $RANKLOG/rank.<N>.out (rank 0 is the one to read)"
mpiexec -n $WORLD -ppn $PPN --cpu-bind depth --depth 16 --no-vni \
    -o "$RANKLOG/rank.%r.out" -e "$RANKLOG/rank.%r.err" \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $DAOS_MNT
RC=$?
DT=$(( $(date +%s) - T0 ))
kill $WATCHDOG_PID 2>/dev/null

CSV=$CKPT_DIR/log_r0.csv
# Count only rows THIS job wrote. CSVLogger APPENDS across jobs sharing
# $CKPT_DIR, and each run writes its own header line -- so the file interleaves
# runs: header, header, 4 rows from the killed 256n job, then this job's header.
#
# mtime-gating does NOT work here (my first attempt): appending refreshes the
# whole file's mtime, so a stale run's rows look current. The header lines are
# the real run delimiter, so count data rows AFTER THE LAST header.
#
# This matters concretely: the 16n validation reported rows=4, loss=0.33922 --
# verbatim the previous run's numbers -- before writing a single iteration.
# (It also explains the "duplicate header" I mistook for a rank-0 race at 3072
# ranks. It was never a race; it is just append mode.)
if [[ -f "$CSV" ]]; then
  ROWS=$(awk -F, '$1=="epoch"{n=0; next} $2 ~ /^[0-9]+$/ {n++} END{print n+0}' "$CSV")
  PRIOR=$(grep -c "^epoch," "$CSV" 2>/dev/null)
  (( PRIOR > 1 )) && echo "NOTE: $CSV has $PRIOR run headers; counting only rows after the last"
else
  ROWS=0
fi
{
  echo "================ DAOS 256n ================"
  echo "job ${PBS_JOBID:-interactive}  $(date)"
  echo "rc=$RC elapsed=${DT}s rows=$ROWS ranks=$WORLD  (NO staging step)"
  [[ -f "$CSV" ]] && { echo "--- CSV ---"; head -1 "$CSV"; sed -n '2p' "$CSV"; tail -2 "$CSV"; }
  # Surface the actual failure instead of guessing. The previous verdict said
  # "check dfuse mount, rendezvous, NA_HOSTUNREACH" when all three were fine and
  # the real cause -- a DAOS agent ping timeout on the head node -- was sitting
  # in the log unmentioned. Rank stdout now lives in $RANKLOG, so grep there too.
  echo "--- failure signals (rank logs + PBS log) ---"
  grep -ahoE "ping RPC timeout from [^ ]+|DistStoreError|NA_HOSTUNREACH|CUDA out of memory|Killed|signal 9" \
       "$RANKLOG"/rank.*.err "$RANKLOG"/rank.*.out 2>/dev/null | sort | uniq -c | sort -rn | head -5
  echo "  (none listed above = no known failure signature)"
  if (( ROWS >= 20 )); then
    echo "VERDICT: PASS -- $ROWS iters at $WORLD ranks reading DAOS."
    echo "  Compare per-iter time against the 16n baseline (~8 s) before prod."
  elif (( ROWS > 0 )); then
    echo "VERDICT: PARTIAL -- reached training ($ROWS iters) then stopped."
    echo "  The data path WORKED; this is a survivability problem, not a scaling one."
    echo "  Read the signals above and $RANKLOG/rank.0.out."
  else
    echo "VERDICT: FAIL -- no iterations. Check the signals above, then"
    echo "  $RANKLOG/rank.0.out for where startup stalled."
  fi
  echo "per-rank logs: $RANKLOG"
} | tee "$VERDICT"
# ---- SELF-RESUBMIT (sustained mode only).
#
# The watchdog kills a hung run; without this, that is where the run ENDS and the
# block is spent. At 256n the fabric desync is expected often enough that a
# single kill would waste the allocation -- the 16n campaign banked e15->e215
# precisely because every kill was followed by a resume.
#
# Guards, mirroring the proven launcher: a .no_relaunch sentinel stops the chain
# by hand; the successor inherits the arm env (a bare qsub would silently resume
# THIS checkpoint dir under DEFAULT settings); and we refuse to resubmit if the
# run banked no epoch, since a config that cannot train once will not train on
# retry either -- that is a loop, not a recovery.
if [[ "$SUSTAINED" == "1" ]]; then
  if [[ -f "$CKPT_DIR/.no_relaunch" ]]; then
    echo "RESUBMIT: .no_relaunch sentinel present -- stopping the chain."
  elif [[ ! -f "$CKPT_DIR/latest.pth.tar" ]]; then
    echo "RESUBMIT: no latest.pth.tar -- the run banked no epoch, so a retry"
    echo "  would repeat the same failure. Investigate before relaunching."
  elif (( $(qstat -u "$USER" 2>/dev/null | grep -c "${VJEPA_JOBTAG:-vg256d}") > 1 )); then
    echo "RESUBMIT: a successor is already queued -- not adding another."
  else
    RESUB_V="VJEPA_SUSTAINED=1"
    [[ -n "${VJEPA_CFG_NAME:-}" ]]   && RESUB_V="$RESUB_V,VJEPA_CFG_NAME=$VJEPA_CFG_NAME"
    [[ -n "${VJEPA_CKPT_DIR:-}" ]]   && RESUB_V="$RESUB_V,VJEPA_CKPT_DIR=$CKPT_DIR"
    [[ -n "${VJEPA_TRUE_ACCUM:-}" ]] && RESUB_V="$RESUB_V,VJEPA_TRUE_ACCUM=$VJEPA_TRUE_ACCUM"
    NEXT=$(qsub -v "$RESUB_V" "$ROOT/scripts/vitG384_256n_daos.sh" 2>&1) \
      && echo "RESUBMITTED: $NEXT (env: $RESUB_V)" || echo "resubmit failed: $NEXT"
  fi
fi

echo "JOB END $(date)"
