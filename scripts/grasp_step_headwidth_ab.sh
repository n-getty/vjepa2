#!/bin/bash
#PBS -N gstep_ab
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

# GraSP STEP head-width / dropout A/B -- ONE arm per invocation.
#
# WHY. The cached Step head-train runs 1.85 s/iter on 1 node x 12 tiles. That is
# NOT I/O and NOT head compute: in job 8775524 the val loop reads the SAME cached
# shards through the same loader at 0.195 s/iter (forward-only). So
# data+H2D+forward <= 11% of the iteration and backward+all-reduce is >= 89%.
# The thing being all-reduced is 3 x 155M-param ASFormer heads (the
# multihead_kwargs LR sweep) at the encoder's full 1664 width = 1.86 GB of fp32
# gradients per iteration, to train a probe at batch_size 2.
#
# `head_embed_dim` (already wired at eval.py:626, precedent in
# configs/heads/grasp/full_cached_384/sweep/hier_fs_e159_probe.yaml) projects
# encoder.embed_dim -> a narrower working width at head entry, so the whole
# O(D^2) body runs narrow:
#     d=1664 -> 466M params, 1862 MB/iter   (current)
#     d= 768 -> 103M params,  412 MB/iter   (4.5x less to reduce)
#
# The same change is also an OVERFITTING fix, which is the real reason to care.
# The d1664 20-epoch run ended at train_acc 97.3% / val_acc 60.3%, with val_loss
# climbing monotonically 1.48 (ep3) -> 3.15 (ep18) while train_loss fell to 0.33.
# A 155M-param head on 9,191 training samples is memorising. Hence the third arm
# pairs the narrow head with dropout 0.3 ([[probe-stability-dropout]]: 0.3 was
# both highest and tightest on cached-384).
#
# THREE SEEDS PER ARM IS NOT OPTIONAL. The GraSP cached-probe single-seed noise
# floor is ~3 mAP ([[grasp-probe-noise-floor-3map]]: two encoders at cos=0.9998
# score 2.91 apart; best.pt-vs-latest.pt on ONE encoder swings 3.17). A 1-seed
# arm cannot distinguish a real effect from noise, so every arm runs s0/s1/s2 and
# the verdict is on the seed MEAN.
#
# TOPOLOGY: 1 node. The head-train is comm-bound, so more ranks make it SLOWER --
# see docs/PROBE_THROUGHPUT_GUIDE.md ("export wide, train narrow"). Do not raise
# select= here.
#
# USAGE (one arm at a time; `debug` allows ONE running + ONE queued job per user):
#   qsub -v ARM=d768_dr00,SEED=0 scripts/grasp_step_headwidth_ab.sh
#
# NB: no `set -u` -- Lmod's init trips it at `module load`
# ([[set-u-module-load-trap]]).
set -o pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
cd "$ROOT"

ARM="${ARM:?must set ARM (d1664_dr00 | d768_dr00 | d768_dr03)}"
SEED="${SEED:-0}"

CFG="$ROOT/configs/heads/grasp/official_ctx16/_ablate/ab_${ARM}_s${SEED}.yaml"
[[ -f "$CFG" ]] || { echo "missing config: $CFG" >&2; exit 2; }

echo "=== gstep_ab ARM=$ARM SEED=$SEED START $(date) PBS_JOBID=$PBS_JOBID ==="
echo "CFG=$CFG"

# eval.py reads the seed from the ENVIRONMENT at import time (eval.py:54), not
# from the YAML -- it must be exported before the trainer starts, and PALS
# forwards the environment to every rank.
export VJEPA_PROBE_SEED="$SEED"
export PROBE_CFG="$CFG"

# Reuse the canonical probe launcher: it derives topology from the PBS
# allocation and carries the settled Aurora CCL/XPU env block. VJEPA_PPN=12 is
# the recorded GraSP protocol (full node).
export VJEPA_PPN=12

bash "$ROOT/scripts/run_asformer_probe_aurora.sh"
RC=$?

echo "=== gstep_ab ARM=$ARM SEED=$SEED DONE $(date) rc=$RC ==="
echo "TIMING: grep 'TIMING\[' the log for the data/head/backward split."
echo "F1:     tail the log_r0.csv under the arm folder."
exit $RC
