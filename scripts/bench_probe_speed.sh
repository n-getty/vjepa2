#!/usr/bin/env bash
# Benchmark probe throughput across (batch_size, use_sdpa) on ONE node, to find
# the OOM ceiling + best iter-time before committing the re-anchor recipe.
#
# Runs each combo for a SHORT slice (a few epochs of the fs10-cached probe OR a
# capped full-data probe), parses per-iter wallclock + peak mem + OOM, prints a
# table. Does NOT keep checkpoints (save_every_iters huge). Read-only wrt the
# real probe runs (writes only to a scratch folder under /tmp + a bench run dir).
#
# Usage (on a held/interactive node, 2 nodes x 12 tiles = 24 ranks):
#   bash scripts/bench_probe_speed.sh <full|cached> [tag]
#
# Sweeps batch_size in BS_LIST x use_sdpa in SDPA_LIST. Override via env.
set -uo pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
MODE="${1:?usage: bench_probe_speed.sh <full|cached> [tag]}"
TAG="${2:-metaraw}"
BS_LIST="${BS_LIST:-2 4 8 12}"
SDPA_LIST="${SDPA_LIST:-true false}"   # cached mode ignores sdpa (no encoder fwd)
BENCH_EPOCHS="${BENCH_EPOCHS:-2}"
SCRATCH=/tmp/probe_bench_${PBS_JOBID:-local}
mkdir -p "$SCRATCH"

# Base config to mutate per combo.
if [[ "$MODE" == "cached" ]]; then
  BASE=$ROOT/configs/heads/sarrarp50/fs10_cached/${TAG}_probe.yaml
  SDPA_LIST="false"   # encoder not run from cache; sdpa irrelevant
else
  BASE=$ROOT/configs/heads/sarrarp50/v2_probe/metaraw_leomatch.yaml
fi
[[ -f "$BASE" ]] || { echo "base config not found: $BASE"; exit 2; }

echo "=== probe speed bench: mode=$MODE base=$(basename $BASE) bs=[$BS_LIST] sdpa=[$SDPA_LIST] epochs=$BENCH_EPOCHS ==="

for sdpa in $SDPA_LIST; do
for bs in $BS_LIST; do
  combo="${MODE}_bs${bs}_sdpa${sdpa}"
  cfg="$SCRATCH/${combo}.yaml"
  # Mutate: batch_size, use_sdpa, num_epochs (short), save_every_iters (off),
  # folder/tag (scratch), and linear-scale head LRs by bs/2.
  $PY - "$BASE" "$cfg" "$bs" "$sdpa" "$BENCH_EPOCHS" "$SCRATCH/$combo" <<'PYEOF'
import sys, yaml, copy
base, out, bs, sdpa, eps, folder = sys.argv[1:7]
bs=int(bs); eps=int(eps); sdpa=(sdpa=="true")
d=yaml.safe_load(open(base))
o=d['experiment']['optimization']
scale=bs/2.0
for h in o.get('multihead_kwargs',[]):
    for k in ('lr','start_lr','final_lr'):
        if k in h: h[k]=h[k]*scale
o['batch_size']=bs
o['num_epochs']=eps
d['model_kwargs']['pretrain_kwargs']['encoder']['use_sdpa']=sdpa
d['save_every_iters']=10**9   # effectively off
d['folder']=folder
d['tag']=f"bench-{bs}-{sdpa}"
d['resume_checkpoint']=False
yaml.safe_dump(d, open(out,'w'), sort_keys=False)
PYEOF
  echo "--- combo=$combo (bs=$bs sdpa=$sdpa) ---"
  t0=$(date +%s)
  # 24 ranks, 2 nodes. Capture OOM / errors.
  if mpiexec --pmi=pmix -n 24 -ppn 12 --cpu-bind depth --depth 16 \
      python -m app.main_dist_aurora --train_mode \
        --fname "$cfg" --params_path "$cfg" \
        > "$SCRATCH/${combo}.log" 2>&1; then
    t1=$(date +%s)
    # parse iters/epoch + epoch span from log timestamps
    its=$(grep -oE "run_one_epoch *\] \[ *[0-9]+\]" "$SCRATCH/${combo}.log" | grep -oE "[0-9]+" | sort -n | tail -1)
    peakmem=$(grep -oE "mem: [0-9.e+]+" "$SCRATCH/${combo}.log" | sed 's/mem: //' | sort -g | tail -1)
    echo "  OK  wall=$((t1-t0))s  max_iter_idx=$its  peak_mem=$peakmem"
  else
    if grep -qiE "out of memory|OOM|alloc.*fail|XPU out" "$SCRATCH/${combo}.log"; then
      echo "  OOM at bs=$bs sdpa=$sdpa  (log: $SCRATCH/${combo}.log)"
    else
      echo "  FAILED (non-OOM) bs=$bs sdpa=$sdpa  -> tail:"; tail -5 "$SCRATCH/${combo}.log"
    fi
  fi
done
done
echo "=== bench done. logs in $SCRATCH ==="
