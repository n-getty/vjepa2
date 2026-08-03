#!/bin/bash
# Measure DAOS read throughput the way TRAINING reads, before trusting it at 256n.
#
# The point is not a synthetic peak number. Staging was replaced because it moved
# 66 TB to deliver 15 TB of reads; the replacement is only better if DAOS sustains
# what the run actually demands. That demand is modest -- 1248 steps x gb 3072 x
# ~4 MB/clip over hours = ~1.5-3 GB/s AGGREGATE, ~6-12 MB/s per node -- so this
# benchmark is really asking "is DAOS within an order of magnitude of that?",
# not "how fast can DAOS go?".
#
# Reads whole .tar shards sequentially, one per rank, which is exactly the
# WebDataset access pattern (streaming a shard start to finish), and reports
# per-rank and aggregate GB/s.
#
#   qsub scripts/daos_read_bench.sh                 # 16 nodes (default)
#   qsub -l select=4 scripts/daos_read_bench.sh     # smaller
#
# NOTE: no `set -u` (memory set-u-module-load-trap).
#
#PBS -N daosrd
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=16
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

POOL=${DAOS_POOL:-AuroraGPT}
CONT=${DAOS_CONT:-vjepa_surg_wds}
MNT=/tmp/${POOL}/${CONT}
PPN=${BENCH_PPN:-12}
# Shards each rank reads. Keep the total modest: we want a rate, not a full pass.
PER_RANK=${BENCH_SHARDS:-4}
NODES=$(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | wc -l); NODES=${NODES:-1}
OUT=/flare/ModCon/ngetty/logs/daos_read_bench_VERDICT.txt

module use /soft/modulefiles
module load daos
# `module load daos` alone leaves no python on PATH -- job 8730443 ran both legs
# straight into "python: command not found" and produced a verdict with headers
# and no measurements. Load frameworks too, and use an absolute interpreter for
# the MPI-launched ranks so a PATH difference on compute nodes cannot repeat it.
module load frameworks
PY=${BENCH_PY:-/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python}
command -v "$PY" >/dev/null 2>&1 || { echo "FATAL: no python at $PY"; exit 1; }

launch-dfuse.sh ${POOL}:${CONT} || { echo "FATAL: launch-dfuse failed"; exit 1; }
mount | grep -q "$CONT" || { echo "FATAL: not mounted at $MNT"; exit 1; }

# Also time the equivalent read straight off Lustre, so the comparison is
# same-hardware / same-hour rather than against a number from a different day.
#
# The Lustre leg MUST be restricted to the same sources the container holds.
# surg_vid_webdataset_resharded has 61 dirs (…_bak_under192, …_staging, …_clean,
# superseded segmentations) against the 16 in the mix, so pointing the benchmark
# at the root would have each leg reading a DIFFERENT set of files at different
# sizes -- not a comparison. Build a symlink farm of exactly what DAOS has.
LUSTRE_ROOT=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
PE_SRC=/flare/ModCon/ngetty/data/pe_video_wds/pe_video

SCRATCH=${SCRATCH_DIR:-/flare/ModCon/ngetty/logs}

# Mirror the container's sources into a symlink farm so the Lustre leg reads the
# SAME set of files. Derived from what the container actually holds, so a partial
# ingest still yields a fair comparison over whatever is present.
LUSTRE_SRC=$SCRATCH/_bench_farm_${PBS_JOBID%%.*}
rm -rf "$LUSTRE_SRC"; mkdir -p "$LUSTRE_SRC"
for s in $(ls -A "$MNT" 2>/dev/null); do
  if [[ -d "$LUSTRE_ROOT/$s" ]]; then
    ln -s "$LUSTRE_ROOT/$s" "$LUSTRE_SRC/$s"
  elif [[ "$s" == "pe_video" && -d "$PE_SRC" ]]; then
    ln -s "$PE_SRC" "$LUSTRE_SRC/pe_video"
  fi
done
echo "Lustre leg mirrors $(ls -A "$LUSTRE_SRC" | wc -l) of $(ls -A "$MNT" | wc -l) container sources"
PYSRC=$SCRATCH/_daos_rd_$$.py
cat > "$PYSRC" <<'PY'
import os, sys, time
root, per_rank, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rank = int(os.environ.get("PALS_RANKID", os.environ.get("PMI_RANK", "0")))
# WORLD_SIZE is exported by the launcher below. Do NOT derive it from PMI/PALS
# vars: they are unreliable across hosts of one multi-host task (memory
# aurora-multi-mpi-per-pbs-worldsize), and a too-small world here would hand
# every rank an overlapping slice -- re-reading cached bytes and reporting a
# flattering number. Fail loudly instead of measuring the wrong thing.
if "WORLD_SIZE" not in os.environ:
    print(f"[rank {rank}] FATAL: WORLD_SIZE unset", flush=True); sys.exit(1)
world = int(os.environ["WORLD_SIZE"])

# Gather candidate shards WITHOUT glob (hangs on dfuse).
shards = []
try:
    for src in sorted(os.listdir(root)):
        d = os.path.join(root, src)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".tar"):
                shards.append(os.path.join(d, f))
except OSError as e:
    print(f"[rank {rank}] LISTING FAILED: {e}", flush=True); sys.exit(1)
if not shards:
    print(f"[rank {rank}] NO SHARDS under {root}", flush=True); sys.exit(1)

mine = shards[rank::max(1, world)][:per_rank]
nbytes = 0
t0 = time.time()
for p in mine:
    try:
        with open(p, "rb") as fh:                 # sequential whole-shard read
            while True:
                b = fh.read(8 << 20)
                if not b:
                    break
                nbytes += len(b)
    except OSError as e:
        print(f"[rank {rank}] READ FAIL {p}: {e}", flush=True)
dt = max(1e-9, time.time() - t0)
print(f"RESULT {tag} rank={rank} files={len(mine)} bytes={nbytes} sec={dt:.2f} "
      f"gbps={nbytes/2**30/dt:.3f}", flush=True)
PY

run () {
  local tag=$1 root=$2
  echo "=== $tag: $((NODES*PPN)) ranks x ${PER_RANK} shards from $root ==="
  WORLD_SIZE=$((NODES * PPN)) \
  mpiexec -n $((NODES * PPN)) -ppn $PPN --cpu-bind none --no-vni \
      --env WORLD_SIZE=$((NODES * PPN)) \
      "$PY" "$PYSRC" "$root" "$PER_RANK" "$tag" 2>&1 | grep -a "^RESULT" \
    > "$SCRATCH/_res_${tag}_$$.txt"
  "$PY" - "$tag" "$SCRATCH/_res_${tag}_$$.txt" <<'PY'
import sys
tag, path = sys.argv[1], sys.argv[2]
tot=n=0; secs=[]
for line in open(path):
    p=dict(kv.split("=",1) for kv in line.split()[2:])
    tot+=int(p["bytes"]); secs.append(float(p["sec"])); n+=1
if not n:
    print(f"  {tag}: NO RESULTS (all ranks failed)"); sys.exit(0)
wall=max(secs); gb=tot/2**30
print(f"  {tag}: {n} ranks, {gb:.1f} GB in {wall:.1f}s wall "
      f"-> {gb/wall:.2f} GB/s aggregate, {gb/wall/n*1024:.1f} MB/s per rank")
PY
  rm -f "$SCRATCH/_res_${tag}_$$.txt"
}

{
  echo "================ DAOS READ BENCH ================"
  echo "job ${PBS_JOBID:-interactive}  $(date)"
  echo "topology ${NODES}n x ${PPN} = $((NODES*PPN)) ranks, ${PER_RANK} shards/rank"
  echo
  run DAOS   "$MNT"
  run LUSTRE "$LUSTRE_SRC"
  echo
  echo "Training demand for reference: ~1.5-3 GB/s aggregate at 256n"
  echo "(1248 steps x gb 3072 x ~4 MB/clip spread over hours)."
  echo "DAOS only has to clear that bar, not win a peak-bandwidth contest."
} | tee "$OUT"

# Fail loudly on a measurement-free run. Job 8730443 emitted a verdict file with
# both headers, no numbers, and exit 0 -- every rank had died on "python: command
# not found". A benchmark that cannot distinguish "measured nothing" from
# "measured something" is worse than no benchmark, because the artifact it leaves
# behind looks like a result.
if ! grep -qE "GB/s aggregate" "$OUT"; then
  echo "BENCH FAILED: no measurements in $OUT -- check the job log for rank errors." | tee -a "$OUT"
  rm -f "$PYSRC"; rm -rf "$LUSTRE_SRC"
  exit 1
fi

rm -f "$PYSRC"; rm -rf "$LUSTRE_SRC"
echo "JOB END $(date)"
