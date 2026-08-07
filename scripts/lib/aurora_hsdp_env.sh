# Shared Aurora env for HSDP + DAOS ViT-G runs. Source it; do not execute it.
#
#     source "$ROOT/scripts/lib/aurora_hsdp_env.sh"
#
# This is the measured recipe from docs/THROUGHPUT_RECIPE_AURORA.md, lifted from
# scripts/vitG384_256n_daos.sh, which was the only launcher carrying all of it.
# It exists because there was no shared file for a recipe change to land in: 25
# ViT-G launchers carried 10 distinct copy-pasted env blocks, several differing
# from each other by exactly one line. That is the drift signature -- the recipe
# improved in one script and the other 24 never heard about it.
#
# *** HSDP ONLY. DO NOT SOURCE THIS INTO A DDP LAUNCHER. ***
# Transport is strategy-dependent, not a global best. HSDP needs
# launcher=none + ofi and NO --pmi=pmix; DDP needs the opposite (pmix/mpi). The
# ofi block below under DDP hung at iter 0 (job 8641936). The vitG1B_capwd_*
# family sets VJEPA_DIST_STRATEGY=ddp and is CORRECT as written -- leave it.
#
# Most values use ${VAR:-default} so `qsub -v FOO=bar` still wins. Three classes
# of deliberate exception, each documented at its line: the two `unset`s
# (CCL_KVS_MODE, LD_PRELOAD -- inheriting either is a bug), and the FI_* block
# (Aurora's own profile exports those, so a :- guard would inherit the SYSTEM
# value and silently discard the recipe -- check `env | grep` in a clean login
# shell before adding a guard to anything new).
#
# NOTE: no `set -u` anywhere in this file or its callers (Lmod init trips it).

# ---- platform
export ZE_FLAT_DEVICE_HIERARCHY=FLAT      # 12 tiles/node addressing
export MPICH_GPU_SUPPORT_ENABLED=1
# HARD-SET, not ${VAR:-default}. Aurora's system profile already exports all
# four of these (FI_PROVIDER="cxi,tcp;ofi_rxm", FI_CXI_RX_MATCH_MODE=hardware,
# FI_CXI_OFLOW_BUF_SIZE=12582912), so a :- guard would inherit the SYSTEM value
# and quietly discard the recipe -- the opposite of what the guard is for. Only
# variables the profile does NOT set can use :-; for these, hard-set is the only
# way the measured value survives, and it matches what every launcher did before
# this fragment existed. Override by editing here, not by qsub -v.
export FI_PROVIDER=cxi
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=131072
export PYTHONFAULTHANDLER=1
# HARD-SET, and it is the same `:-` trap as the FI_* block above: this line used
# to read ${TMPDIR:-/tmp}, and PBS ALWAYS sets TMPDIR, so the guard never fired
# and the intended /tmp was never applied. Required for num_workers>0:
#
#   PBS TMPDIR              /var/tmp/pbs.<jobid>.<server>       68 chars
#   + PALS per-launch uuid  .../<uuid>/tmp                     109   (+41)
#   + /pymp-XXXXXXXX/listener-XXXXXXXX                         141   cap 107
#
# AF_UNIX sun_path is capped at 108 incl. NUL, so python multiprocessing's
# resource_sharer listener cannot bind: `OSError: AF_UNIX path too long`. It is
# raised on the non-fatal queue feeder THREAD, so the loader never delivers a
# batch and the run HANGS rather than erroring -- that is symptom (B), job
# 8740716's n1_nw2_prof rung, 0 iterations. Measured from inside the ranks in
# job 8740830; do NOT re-derive it from a login or job shell, both of which
# miss the UUID and report a passing ~102.
#
# TMPDIR headroom is 107-32 = 75 chars. /tmp leaves 71 spare even after PALS.
# ALCF documents this (user-guides aurora/known-issues.md #7, "Set TMPDIR to
# avoid AF_UNIX path too long"), and all four BaseMM_PRISM Aurora launchers
# already export TMPDIR=/tmp. /tmp on Aurora compute is a 504 G node-local
# tmpfs, so nothing is lost by moving off the PBS dir.
export TMPDIR=/tmp
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"

# ---- oneCCL transport (HSDP)
export CCL_PROCESS_LAUNCHER=${CCL_PROCESS_LAUNCHER:-none}
export CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-ofi}
export CCL_KVS_IFACE=${CCL_KVS_IFACE:-hsn0}
# UNSET, not empty. oneCCL validates this enum and rejects '': it raised
# "CCL_KVS_MODE: unexpected value: , expected values: pmi, mpi, pmix_ofi,
# pmix_ofi_shm" and killed the accum=2 arm of job 8731004 at iter 0. findings 5a
# literally recommends `CCL_KVS_MODE=` (empty) -- that guidance is wrong for this
# build. These two are DDP-oriented globals that conflict with oneCCL bringing up
# its own KVS over CXI, which launcher=none + ofi requires; removing them is what
# neutralizes them. Not ${VAR:-} guarded on purpose: inheriting either is a bug.
unset CCL_KVS_MODE CCL_KVS_USE_MPI_RANKS
export CCL_OP_SYNC=${CCL_OP_SYNC:-1}          # =0 is fatal: wedge from iter 0 (8641553)
export CCL_WORKER_COUNT=${CCL_WORKER_COUNT:-1} # =4 stalls the first batch even at 1n
# ring / 16 MB re-validated at 64n (job 8732160): double_tree 1.00x, 64 MB 0.97x,
# IQRs overlap everywhere. ring-vs-tree does NOT flip at 63 hops.
export CCL_ALLREDUCE=${CCL_ALLREDUCE:-ring}
export CCL_CHUNK_SIZE=${CCL_CHUNK_SIZE:-16777216}
# CCL_WARN emitted 10,900 lines at 3072 ranks (expected device-uuid warnings
# under HSDP). Not actionable, and a tenth of the log funnel that killed 8730678.
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-error}

# ---- distribution strategy
export VJEPA_DIST_STRATEGY=${VJEPA_DIST_STRATEGY:-hsdp}
export FSDP_SHARDING=${FSDP_SHARDING:-shard_grad_op}
# 3600 s. 192 ranks needed >300 s to rendezvous; 3072 ranks is 16x that. The
# in-code default is 900 (src/utils/distributed.py) -- this raises it further for
# the scale-out path specifically.
export TORCH_DIST_TIMEOUT_SECONDS=${TORCH_DIST_TIMEOUT_SECONDS:-3600}

# ---- data path
# 0 = global-rank slicing. MUST be 0 on DAOS. The flag exists only because each
# node's /tmp held a DIFFERENT shard subset; on DAOS every node sees the whole
# corpus, so local slicing makes all N nodes compute the SAME 12 slices -- Nx
# data duplication with no error message.
export WDS_LOCAL_SLICING=${WDS_LOCAL_SLICING:-0}
# 2, not 0. The old comment here said "WebDataset streaming is I/O-light; workers
# cost memory and socket pressure" -- measured at 64n (job 8741170, paired arms,
# one allocation, 768/768 rank CSVs both) that is wrong by 3.45x:
#
#   nw=0: mean 24.65 s/iter, dataload median 20.18 s, ALL 23/23 iters stalled
#   nw=2: mean  7.15 s/iter, dataload median  0.00 s,      8/23 iters stalled
#
# The compute floor moves only +0.22 s between them, so this is purely loading.
# With nw=0 every rank decodes inline on the critical path; the cost is real,
# not merely moved into a visible column.
#
# Two prerequisites, both landed -- do NOT raise this above 0 on a tree missing
# either, or it hangs or throws at teardown:
#   1. TMPDIR=/tmp (a31e1f1). Worker AF_UNIX socket paths must fit sun_path's
#      107-byte cap; the default PBS TMPDIR is 141 chars and overflows it.
#   2. The loader-destructor exit fix (281b3fe) + persistent_workers.
# The xccl-fork deadlock is O(ranks) and was the last open risk; n64_nw2 cleared
# it at 768 ranks with no `terminate called`.
export VJEPA_NUM_WORKERS=${VJEPA_NUM_WORKERS:-2}
# Declared rather than inherited: it is a -3 GB memory lever (not a speed lever)
# and the code default is already 1, so an outer env setting 0 would silently
# lose it with no log line.
export VJEPA_USE_XPU_FLASH=${VJEPA_USE_XPU_FLASH:-1}
# libpil4dfs hangs FSDP AllGather (DAOS-17499). Never inherit it.
unset LD_PRELOAD

# Reminder for the caller, since these cannot live in an env file:
#   mpiexec ... --no-vni            (DAOS RPCs fail NA_HOSTUNREACH without it)
#           ... -o "$D/rank.%r.out" -e "$D/rank.%r.err"   (never funnel stdout)
#           ... and NO --pmi=pmix   (that is the DDP transport)
