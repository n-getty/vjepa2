#!/usr/bin/env bash
# HOLD-NODE PROBE: activation-checkpointing OFF under HSDP, 1 node.
# Question 1: does the full (un-recomputed) 48-layer 2B activation set fit in the
#             ~42GB headroom HSDP freed (baseline HSDP+ckpt-on was 22.0GB)?
# Question 2: how much faster is iter time without the ~20-30% recompute tax?
# Baseline for comparison (same SMOKE_vitG384, HSDP, ckpt ON): iter 7.16s, mem 22.0GB.
# DDP+ckpt-on (the old default) was iter 7.93s, mem 57.6GB.
#
# NOT a batch-size or image-arm change — ONLY use_activation_checkpointing:false.
# Output is bit-identical to ckpt-on (recompute only affects backward memory, not
# math), so this is pure throughput with zero recipe risk.
#
# Run by dropping this into a hold node's CMD_DIR as run_N.sh, OR qsub directly.
# Intended: staged for the hold-node workflow after the 16n spike passes.
set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
cd $ROOT
PY_STAGE=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
BASE_CFG=$ROOT/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
RUNTIME_CFG=$ROOT/.runtime_configs/g12_weak/configs/vitg16_surg_vid_webdataset_single4/SMOKE_vitG384.yaml
CKPT_DIR=/flare/ModCon/ngetty/checkpoints/SMOKE_vitG384/hsdp_actckptoff_n1g12
PARAMS=$CKPT_DIR/params-pretrain.yaml
mkdir -p $CKPT_DIR

# Stage runtime cfg, then patch: fresh folder, no-save, AND activation ckpt OFF.
$PY_STAGE $ROOT/scripts/prepare_runtime_config.py $BASE_CFG --root $ROOT --num-gpus 12 --num-nodes 1 --weak-scale > /dev/null
$PY_STAGE - "$RUNTIME_CFG" "$PARAMS" "$CKPT_DIR" <<'PY'
import sys, yaml
src, dst, folder = sys.argv[1], sys.argv[2], sys.argv[3]
d = yaml.safe_load(open(src))
d["folder"] = folder + "/fresh"
d["meta"]["save_every_freq"] = 10**9
d["model"]["use_activation_checkpointing"] = False   # <-- the only change under test
yaml.safe_dump(d, open(dst, "w"), sort_keys=False)
print(f"patched -> {dst}: use_activation_checkpointing=False, folder={d['folder']}")
PY
mkdir -p $CKPT_DIR/fresh

export VJEPA_DIST_STRATEGY=hsdp LOCAL_WORLD_SIZE=12 FSDP_SHARDING=shard_grad_op
unset VJEPA_BF16_COMM
export WORLD_SIZE=12
export LOCAL_DATA_ROOT=/tmp/vjepa_data/hsdp_smoke_hold   # reuse staged smoke shards if present
if [ ! -d "$LOCAL_DATA_ROOT" ]; then
  echo "=== staging smoke datasets (first use) ==="
  mpiexec -n 1 -ppn 1 --cpu-bind none \
      python $ROOT/scripts/stage_node_shards.py --params $PARAMS \
      --local-root $LOCAL_DATA_ROOT --num-nodes 1 --local-world-size 12 --workers 8 2>&1 | tail -3
fi

echo "=== HSDP + activation-checkpointing OFF: 12-rank train ==="
timeout 700 mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
    python -m app.main_dist_aurora --train_mode \
        --fname $PARAMS --params_path $PARAMS \
        --local_data_root $LOCAL_DATA_ROOT 2>&1 \
  | grep -viE "UserWarning|warnings.warn|FutureWarning|comm_dev_uuids|CCL_WARN" \
  | grep -iE "wrapped|mem:|loss:|Epoch [0-9]|Error|Traceback|banned|UR_RESULT|OUT_OF" | tail -40
echo "actckpt-off train rc=${PIPESTATUS[0]}"
echo "COMPARE: HSDP+ckpt-ON was iter 7.16s / mem 22.0GB ; DDP+ckpt-ON was iter 7.93s / mem 57.6GB"
