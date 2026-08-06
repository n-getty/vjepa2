#!/bin/bash
# PAIRED CCL KNOB SWEEP at the target scale.
#
# WHY THIS EXISTS
# ---------------
# Every CCL tuning decision in this repo was measured at 16 nodes, and the
# launcher says so about itself (app/main_dist_aurora.py:154-157, on rejecting
# rabenseifner): "intra-node fanout overhead exceeds inter-node savings AT ONLY
# 16 NODES -- ALCF's large scale recs target 64+ nodes."
#
# Under HSDP the shard dim is intra-node (12 tiles) and the replicate dim is
# INTER-node, so the allreduce spans NODE COUNT:
#
#     nodes | ring hops | double_tree levels
#        16 |        15 | ~4     <- where every decision was made
#        64 |        63 | ~6
#       256 |       255 | ~8
#
# Ring is O(P), double_tree O(log P). At 15 hops ring's lower constant wins --
# that is exactly what jobs 8645758/8645804 measured. At 255 hops, 17x the chain
# depth, the crossover may flip. The same A/B was also read as REFUTING chain
# depth as the spike driver, but it refuted it where chain depth is cheap by
# construction; that does not transfer.
#
# ARMS (add/remove via SWEEP; each is one env var against a common baseline)
#   ring16   CCL_ALLREDUCE=ring        CCL_CHUNK_SIZE=16M   <- current production
#   tree16   CCL_ALLREDUCE=double_tree CCL_CHUNK_SIZE=16M
#   ring64   CCL_ALLREDUCE=ring        CCL_CHUNK_SIZE=64M
#
# CCL_CHUNK_SIZE has NEVER been A/B'd anywhere. findings 4f calls it "a
# first-class variable" under HSDP and "a low-cost A/B"; the 16 MiB value was
# chosen for the DDP workload.
#
# DESIGN
#   - all arms in ONE allocation, back to back, same nodes. Iter-time CV is ~81%
#     at 64n; across separate jobs the fabric-hour term would swamp the effect.
#   - the CURRENT PRODUCTION setting runs FIRST, so it gets the cold cache and
#     any challenger has to beat a warmed machine. Conservative direction.
#   - WALL-CLOCK iter time only. Not backward-ms: PhaseTimer XPU event deltas go
#     NEGATIVE on stalled collectives (-139s/-326s/-277s at 64n job 8730919),
#     i.e. exactly the iterations a comms sweep is about.
#   - the report refuses to declare a winner when IQRs overlap.
#
#   qsub -l select=256 scripts/ccl_knob_sweep.sh      # the scale that matters
#   qsub -l select=64  scripts/ccl_knob_sweep.sh      # cheaper dry run
#
# NOTE: no `set -u` (memory set-u-module-load-trap).
#
#PBS -N cclsweep
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
MODELS_CONT=${DAOS_MODELS_CONT:-vjepa_models}
DAOS_MNT=/tmp/${POOL}/${CONT}
MODELS_MNT=/tmp/${POOL}/${MODELS_CONT}
PPN=12
NNODES=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l); NNODES=${NNODES:-256}
WORLD=$(( NNODES * PPN ))
# Per arm. Keep short: 3 arms x IPE x ~10 s/iter must fit the hour WITH ~10 min
# of startup (rendezvous + checkpoint load) paid ONCE per arm.
IPE=${SWEEP_IPE:-25}
SWEEP=${SWEEP_ARMS:-"ring:16777216 double_tree:16777216 ring:67108864"}
CFG_NAME=${VJEPA_CFG_NAME:-vitG384_lbA8}
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
TAG="${PBS_JOBID%%.*}"; TAG="${TAG:-manual}"
OUTROOT=/flare/ModCon/ngetty/checkpoints/ccl_sweep/${TAG}
VERDICT=/flare/ModCon/ngetty/logs/ccl_sweep_VERDICT.txt
mkdir -p "$OUTROOT" /flare/ModCon/ngetty/logs

echo "JOB START $(date) PBS_JOBID=$PBS_JOBID  ${NNODES}n x ${PPN} = ${WORLD} ranks"
echo "arms: $SWEEP   ipe=$IPE each"
echo "inter-node allreduce spans ${NNODES} ranks -> ring $(( NNODES - 1 )) hops"

cd $ROOT
module use /soft/modulefiles
module load frameworks
module load daos
export PYTHONNOUSERSITE=1
source /flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/activate
export PYTHONPATH=$ROOT:$PYTHONPATH
# The measured HSDP/DAOS recipe (CCL transport, FI, WDS_LOCAL_SLICING=0,
# the LD_PRELOAD and CCL_KVS_MODE unsets). Overrides go BELOW the source.
source $ROOT/scripts/lib/aurora_hsdp_env.sh
# The fragment sets CCL_ALLREDUCE=ring / CCL_CHUNK_SIZE=16M -- the production
# values, which are also the ring16 arm. That is NOT a confound: run_arm passes
# every arm's pair via `mpiexec --env`, which wins over the inherited value, so
# each arm gets exactly the algo/chunk its name says. The inherited pair only
# ever applies if an arm forgets to specify one.
export LOCAL_WORLD_SIZE=$PPN
export VJEPA_TRUE_ACCUM=${VJEPA_TRUE_ACCUM:-1}
# Deliberately NOT set (findings 4e, env-diff job 8643398 falsified all three:
# run still spiked cohort-wide, l0-free/l0-ext flat within +-5 MiB over 74 iters):
#   CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD, FI_MR_CACHE_MONITOR, PYTORCH_ALLOC_CONF
# findings 5a re-adds them; 4e measured them useless. Trust the experiment.
unset CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD FI_MR_CACHE_MONITOR PYTORCH_ALLOC_CONF

if [[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]]; then
  MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
else
  MASTER_ADDR=$(hostname)
fi
export MASTER_ADDR MASTER_PORT=29500 WORLD_SIZE=$WORLD

timeout 600 launch-dfuse.sh ${POOL}:${MODELS_CONT} || { echo "FATAL: launch-dfuse models"; exit 1; }
timeout 600 launch-dfuse.sh ${POOL}:${CONT}        || { echo "FATAL: launch-dfuse corpus"; exit 1; }
timeout 60 ls "$DAOS_MNT"   >/dev/null 2>&1 || { echo "FATAL: $DAOS_MNT unresponsive"; exit 1; }
timeout 60 ls "$MODELS_MNT" >/dev/null 2>&1 || { echo "FATAL: $MODELS_MNT unresponsive"; exit 1; }
echo "DAOS mounted (corpus + models)"

$PY $ROOT/scripts/prepare_runtime_config.py \
  $BASE_CFG --root $ROOT --num-gpus $PPN --num-nodes $NNODES --weak-scale > /dev/null || exit 1
RUNTIME_CFG=$ROOT/.runtime_configs/n${NNODES}g${PPN}_weak/configs/vitg16_surg_vid_webdataset_single4/${CFG_NAME}.yaml

run_arm () {
  local algo=$1 chunk=$2
  local name="${algo}_$(( chunk / 1048576 ))M"
  local dir=$OUTROOT/$name
  local params=$dir/params.yaml
  mkdir -p "$dir"
  cp "$RUNTIME_CFG" "$params" || return 1
  $PY - "$params" "$dir" "$IPE" "$MODELS_MNT" <<'PY'
import sys, yaml, os
p, d, ipe, models = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
c = yaml.safe_load(open(p))
c["folder"] = d
c["optimization"]["ipe"] = ipe
c["optimization"]["epochs"] = 1
meta = c.setdefault("meta", {})
ck = meta.get("pretrain_checkpoint")
if ck:
    cand = os.path.join(models, os.path.basename(ck))
    if not os.path.exists(cand):
        sys.exit(f"FATAL: {cand} missing")
    meta["pretrain_checkpoint"] = cand
yaml.safe_dump(c, open(p, "w"), sort_keys=False)
PY
  echo "===== ARM $name (CCL_ALLREDUCE=$algo CCL_CHUNK_SIZE=$chunk) ====="
  local t0=$(date +%s)
  mpiexec -n $WORLD -ppn $PPN --cpu-bind depth --depth 16 --no-vni \
      --env CCL_ALLREDUCE=$algo --env CCL_CHUNK_SIZE=$chunk \
      -o "$dir/rank.%r.out" -e "$dir/rank.%r.err" \
      python -m app.main_dist_aurora --train_mode \
          --fname $params --params_path $params \
          --local_data_root $DAOS_MNT
  echo "  arm $name rc=$? in $(( $(date +%s) - t0 ))s"
}

for spec in $SWEEP; do
  run_arm "${spec%%:*}" "${spec##*:}"
done

$PY - "$OUTROOT" "$NNODES" <<'PY' | tee "$VERDICT"
import os, sys, statistics
root, nodes = sys.argv[1], int(sys.argv[2])
print("================ CCL KNOB SWEEP ================")
print(f"root {root}   {nodes} nodes -> ring {nodes-1} hops/allreduce\n")
res = {}
for name in sorted(os.listdir(root)):
    csv = os.path.join(root, name, "log_r0.csv")
    if not os.path.isfile(csv):
        continue
    it = []
    for line in open(csv):
        p = line.split(",")
        if p and p[0] == "epoch":
            it = []; continue
        if len(p) > 3 and p[1].strip().isdigit() and int(p[1]) >= 5:
            it.append(float(p[3]) / 1000.0)
    if not it:
        print(f"  {name:16s} no iters past warmup"); continue
    s = sorted(it)
    res[name] = dict(n=len(s), med=statistics.median(s),
                     p25=s[len(s)//4], p75=s[3*len(s)//4], lo=s[0], hi=s[-1])
for name, r in res.items():
    print(f"  {name:16s} n={r['n']:3d}  median {r['med']:6.2f}s  "
          f"IQR {r['p25']:5.1f}..{r['p75']:<6.1f} range {r['lo']:.1f}..{r['hi']:.1f}")
if len(res) > 1:
    base = res.get("ring_16M")
    if base:
        print(f"\n  vs ring_16M (current production, ran first / cold):")
        for name, r in res.items():
            if name == "ring_16M":
                continue
            spd = base["med"] / r["med"]
            ov = not (base["p75"] < r["p25"] or r["p75"] < base["p25"])
            verdict = "IQRs OVERLAP -- unresolved" if ov else "resolved"
            print(f"    {name:16s} {spd:5.2f}x   ({verdict})")
        print("\n  An overlapping IQR means this sample does not separate the arms.")
        print("  Do not report a winner from it -- lengthen the arms and repeat.")
print("\n  Wall-clock iter time for the verdict above. backward-ms IS usable if you")
print("  unwrap it: a negative XPU event delta is a 32-bit counter rollover")
print("  (80 ns tick -> 343597.38368 ms period), not a broken timer. Add one")
print("  period. See scripts/scaling_efficiency.py:unwrap and the measurement")
print("  notes in docs/THROUGHPUT_RECIPE_AURORA.md (verified 2026-08-06).")
PY
echo "JOB END $(date)"
