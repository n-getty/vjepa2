#!/bin/bash
# ONE-TIME: copy the WebDataset corpus from Lustre into the DAOS container
# AuroraGPT/vjepa_surg_wds, so training reads from DAOS instead of staging to /tmp.
#
# WHY THIS EXISTS
# ---------------
# Per-node /tmp staging does not survive scale-out. Measured on a real 16n run
# (job 8729208): 0.43 GB/s/node, ~6.9 GB/s aggregate off Lustre. At 256 nodes the
# corpus at shard-floor 48 is 265 GB/node = 66 TB aggregate -- ~10 min if per-node
# bandwidth scales, ~2.7 HOURS if Lustre saturates near that aggregate. Either way
# it is 66 TB of copying to deliver 15 TB of reads (each node stages 265 GB but the
# whole run only reads ~58 GB/node): we move 4.6x more than we ever touch.
#
# DAOS removes the staging step entirely. The corpus lives once in the container
# (4.5 TB against 259 TB free in the AuroraGPT pool) and every node reads it
# directly over the fabric. The run's actual streaming demand is only ~1.5-3 GB/s
# aggregate -- it spreads 15 TB over hours instead of minutes -- which is well
# inside what DAOS delivers, and there is no per-job copy at all.
#
# Also removes two second-order problems: the shard-floor tradeoff (per-worker
# variety no longer costs disk) and the 4.6x write amplification.
#
# PREREQ: container already created (done 2026-08-03):
#   daos container create --type=POSIX --chunk-size=2097152 \
#     --properties=rd_fac:3,ec_cell_sz:131072,cksum:crc32,srv_cksum:on \
#     --file-oclass=EC_16P3G32 --dir-oclass=RP_4G1 AuroraGPT vjepa_surg_wds
#
# NOTE the oclass: G32, not the GX that BaseMM_PRISM's scripts use. Current ALCF
# guidance (user-guides .../daos/daos-overview.md:467) is that GX stripes each
# file across ALL servers -- optimal for ONE big shared file -- while G32 stripes
# across 32, which is what you want for many independent files read by many
# ranks. A WebDataset corpus is thousands of independent .tar shards, so G32 is
# the right side of that tradeoff. PRISM's scripts predate the guidance change
# (they were copied from an older revision of the same doc); do not copy GX from
# them without re-reading that section.
#
#   qsub scripts/daos_ingest_corpus.sh
#
# Idempotent: dsync only transfers differences, so a re-run after a partial or
# interrupted copy resumes rather than starting over.
#
# NOTE: no `set -u` (memory set-u-module-load-trap).
#
#PBS -N daosing
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=8
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

POOL=${DAOS_POOL:-AuroraGPT}
CONT=${DAOS_CONT:-vjepa_surg_wds}
SRC_ROOT=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
PE_SRC=/flare/ModCon/ngetty/data/pe_video_wds/pe_video
MNT=/tmp/${POOL}/${CONT}          # launch-dfuse.sh mounts here on every node
PPN=${INGEST_PPN:-12}
NODES=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l); NODES=${NODES:-1}
VERDICT=/flare/ModCon/ngetty/logs/daos_ingest_VERDICT.txt

echo "JOB START $(date) PBS_JOBID=$PBS_JOBID  nodes=$NODES ppn=$PPN"

module use /soft/modulefiles
module load daos
module load mpifileutils

# Mount the container on EVERY node in the allocation (clush wrapper).
launch-dfuse.sh ${POOL}:${CONT} || { echo "FATAL: launch-dfuse failed"; exit 1; }
mount | grep -q "$CONT" || { echo "FATAL: container not mounted at $MNT"; exit 1; }
echo "mounted $POOL:$CONT at $MNT"

# The 16 sources of the live corpus. pe_video lives outside SRC_ROOT and is a
# symlink tree, so dsync must dereference (-L) or it copies dangling links.
SOURCES=(small_surg sitl surgenet_robotic_clean surgtoolloc2022 surgvu24_clean
         grasp_noleak cholec80 sitl_2026 lemon heichole_512 multibypass140
         gynsurg lapgyn6_events surgenet_lap openh)

T0=$(date +%s)
FAILED=0

# --dereference on EVERY source, not just the ones known to be symlink trees.
# The first attempt (job 8730269) only dereferenced pe_video and small_surg came
# through as 413 bytes of DANGLING symlinks: it is a bundle of links into
# ../crcd/, ../endovis15/ etc., and those relative targets do not exist inside
# the container. That is a silent corpus corruption -- dsync exits 0, the tar
# COUNT matches, and training only discovers it when a shard fails to open.
# Currently small_surg and pe_video are the symlink trees, but deref is a no-op
# for regular files, so applying it everywhere removes the whole class of bug
# rather than maintaining a list that will drift.
copy_one () {
  local name=$1 src=$2
  [[ -d "$src" ]] || { echo "SKIP $name (missing)"; return 0; }
  echo "=== $name ==="
  local t=$(date +%s)
  mpiexec -n $((NODES * PPN)) -ppn $PPN --cpu-bind none --no-vni \
      dsync --progress 30 --bufsize 64MB --dereference "$src" "$MNT/$name" \
    || { echo "  DSYNC FAILED for $name"; FAILED=1; }
  echo "  $name done in $(( $(date +%s) - t ))s"
}

for s in "${SOURCES[@]}"; do copy_one "$s" "$SRC_ROOT/$s"; done
copy_one pe_video "$PE_SRC"

DT=$(( $(date +%s) - T0 ))

# Verify: every source must have its metadata.json and a plausible .tar count.
# A silently short copy is the failure that would poison training later, so this
# compares against the Lustre original rather than just checking for existence.
{
  echo "================ DAOS INGEST ================"
  echo "job ${PBS_JOBID:-interactive}  $(date)"
  echo "elapsed ${DT}s on ${NODES} nodes x ${PPN} ranks"
  echo
  bad=0
  for s in "${SOURCES[@]}" pe_video; do
    d=$MNT/$s
    [[ -d "$d" ]] || { echo "  MISSING  $s"; bad=1; continue; }
    src=$SRC_ROOT/$s; [[ "$s" == "pe_video" ]] && src=$PE_SRC
    [[ -d "$src" ]] || continue
    n_dst=$(ls "$d"/*.tar 2>/dev/null | wc -l)
    n_src=$(ls "$src"/*.tar 2>/dev/null | wc -l)
    meta=$([[ -f "$d/metadata.json" ]] && echo yes || echo NO)
    # BYTES, not just counts. The symlink failure produced a matching tar count
    # with 413 bytes of dangling links behind it -- counts alone call that "ok".
    # -L follows links on the source so we compare real bytes to real bytes.
    b_dst=$(du -sbL "$d" 2>/dev/null | cut -f1); b_dst=${b_dst:-0}
    b_src=$(du -sbL "$src" 2>/dev/null | cut -f1); b_src=${b_src:-1}
    # Any surviving symlink in the destination is a copy that did not dereference.
    n_link=$(find "$d" -maxdepth 1 -type l 2>/dev/null | wc -l)
    pct=$(( 100 * b_dst / (b_src > 0 ? b_src : 1) ))
    if [[ "$n_dst" -ne "$n_src" || "$meta" == "NO" ]]; then
      echo "  MISMATCH $s: tars $n_dst/$n_src metadata=$meta"; bad=1
    elif (( n_link > 0 )); then
      echo "  SYMLINKS $s: $n_link unresolved links -- re-run with --dereference"; bad=1
    elif (( pct < 99 )); then
      echo "  SHORT    $s: ${pct}% of source bytes ($b_dst/$b_src)"; bad=1
    else
      echo "  ok       $s: $n_dst tars, ${pct}% bytes, metadata present"
    fi
  done
  echo
  if (( bad || FAILED )); then
    echo "INGEST VERDICT: INCOMPLETE -- re-run this job (dsync resumes)."
  else
    echo "INGEST VERDICT: COMPLETE. Point configs at $MNT/<source> and drop staging."
  fi
} | tee "$VERDICT"

echo "JOB END $(date)"
