#!/bin/bash
#PBS -N gopenbuf
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare:daos_user_fs
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
#
# ONE NODE, READ-ONLY: is DAOS tar read granularity the owner of the dataload tail?
#
# See scripts/probe_gopen_buffer.py's header for the mechanism. Short version:
# webdataset opens shards via gopen -> open(url, "rb", buffering=int($GOPEN_BUFFER
# or -1)), and tarfile stream mode requests 10240 B per read, so the syscall size
# hitting the filesystem is whatever `buffering` resolves to. At -1 CPython uses
# st_blksize. If dfuse reports 4 KB, a 1.5 GB shard is ~190 k read RPCs through
# the DAOS client instead of ~1.5 k at 1 MB.
#
# WHY IT NEEDS A COMPUTE NODE. st_blksize and read cost are properties of the
# MOUNT. On a login node the DAOS container is not mounted at all, so the probe
# can only reach Lustre -- where it already self-refutes (st_blksize is 4 MB
# there and the curve is flat, measured 2026-08-07). The DAOS column is the
# entire reason for this job.
#
# WHY THE debug QUEUE AND NOT debug-scaling. Both are max_run=1 per user and
# job 8741170 (the 64n nw=2 hazard ladder) already holds debug-scaling. This is
# 1 node and read-only, so it fits debug and costs that pending job nothing.
#
# Submit:  qsub -A AuroraGPT -q debug scripts/probe_gopen_buffer_pbs.sh
#
# This job trains nothing, writes nothing to the corpus, and mounts the DAOS
# container read-only.

set -o pipefail
ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PY=/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/python
POOL=${DAOS_POOL:-AuroraGPT}
CONT=${DAOS_CONT:-vjepa_surg_wds}
export DAOS_MNT=/tmp/${POOL}/${CONT}
cd "$ROOT" || exit 1

# Never `set -u` in an Aurora PBS launcher -- Lmod's init trips it.
module use /soft/modulefiles
module load frameworks
module load daos

timeout 600 launch-dfuse.sh ${POOL}:${CONT} || { echo "FATAL: launch-dfuse"; exit 1; }
timeout 60 ls "$DAOS_MNT" >/dev/null 2>&1 || { echo "FATAL: $DAOS_MNT unresponsive"; exit 1; }
echo "DAOS mounted at $DAOS_MNT"
echo

# Two sources on purpose, and they are a matched pair: surgvu24_clean and its
# re-encoded twin surgvu24_clean_g16 hold the same clips at different member
# sizes. If a buffer effect exists it must not care which of these it is -- read
# granularity is a property of the FILE, not the codec -- so an effect that
# shows up in one and not the other is a reason to distrust it rather than to
# report it. Both are live training sources with 2000 shards each, so neither
# can silently skip for want of shards (an earlier draft named `pe_video`, which
# is not under this Lustre root at all and would have skipped in exactly that
# way -- printing a SKIP line that reads like a minor note in a long log).
for SRC in surgvu24_clean_g16 surgvu24_clean; do
  echo "##################### source=$SRC #####################"
  timeout 1200 "$PY" scripts/probe_gopen_buffer.py --source "$SRC" \
      --members 24 --repeats 3 --require-daos
  rc=$?
  [ $rc -eq 0 ] || echo "!!! $SRC exited rc=$rc -- see the FATAL line above"
  echo
done

echo "JOB DONE $(date)"
