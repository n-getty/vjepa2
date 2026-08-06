#!/usr/bin/env python
"""Minimal repro/bisect for the nw>0 exit throw on Aurora XPU.

THE SYMPTOM
-----------
With num_workers=2, 85 of 96 ranks print at process exit, AFTER the last CSV row
and AFTER the checkpoint:

    terminate called after throwing an instance of 'std::system_error'
      what():  No such file or directory

Nothing is lost -- but the exit status is destroyed, and VJEPA_SUSTAINED
self-resubmit branches on exit status. It is nw-specific: 0 of 300 ranks across
three nw=0 rungs in the same and adjacent jobs throw.

WHY A MINIMAL REPRO, RATHER THAN A GDB RUNG ON THE REAL TRAINER
---------------------------------------------------------------
The real stack takes ~5 min of startup and a 22.8 GB checkpoint read before it
can throw, and it stacks four candidate layers on top of each other. This runs
in seconds and turns them on one at a time, so the answer is "which layer", not
"somewhere in the trainer".

WHAT IS ALREADY RULED OUT -- do not re-propose these as the fix:
  * MP_SOCKET_DIR=/tmp                        (main_dist_aurora.py:39, active)
  * set_sharing_strategy("file_system")       (main_dist_aurora.py:358, active)
  * persistent_workers / the loader keepalive (98ca1ce -- fixed the STALL; the
    throw survived it, which is exactly how we know they are separate defects)

AND ONE CONFOUND THAT MUST BE TESTED FIRST, hence stage `sigterm`:
main_dist_aurora.py:443-449 records rank 0 throwing this SAME exception on an
ordinary PBS SIGTERM, with no DataLoader involved. If plain SIGTERM reproduces
it, the throw is what a signal-kill looks like on this stack generally and the
nw correlation is about which ranks get signalled, not about workers. That would
make the whole "forked workers" framing wrong, so it is stage one.

STAGES (--stage), each adds exactly one layer:
  bare     spawn DataLoader workers, CPU tensors only. No XPU, no distributed.
  xpu      + pin an XPU device and move batches to it.
  dist     + init_process_group(xccl). This is the real trainer's shape.
  sigterm  no DataLoader at all; raise SIGTERM at the end. The confound test.

Run all four, in order, and the first one that throws names the layer.

USAGE (12 ranks, one node):
    mpiexec -n 12 -ppn 12 --cpu-bind depth --depth 16 --no-vni \\
        python scripts/repro_nw_exit_throw.py --stage dist --num-workers 2

Set VJEPA_HARD_EXIT=0 in the caller's environment for the real trainer; this
script never calls os._exit, so destructors always run -- that is the point.
faulthandler is armed so a fatal signal dumps Python frames, and the C++ frame
comes from the core file (`ulimit -c unlimited`).
"""

import argparse
import faulthandler
import os
import signal
import sys
import time

# Arm BEFORE torch is imported, so a crash during import is still caught.
faulthandler.enable(all_threads=True)
# SIGABRT is what `terminate called after throwing` raises. Without this the
# Python side of the stack is lost and only the C++ message survives.
for _sig in ("SIGABRT", "SIGSEGV", "SIGBUS", "SIGFPE"):
    try:
        faulthandler.register(getattr(signal, _sig), all_threads=True, chain=True)
    except (AttributeError, ValueError):
        pass

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


class TinyDataset(Dataset):
    """Deliberately trivial. The subject is worker teardown, not decode -- any
    real decode work would add ffmpeg/decord destructors to the suspect list."""

    def __init__(self, n=512):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.full((3, 8, 64, 64), float(i % 7))


def rank_env():
    r = int(os.environ.get("PALS_RANKID", os.environ.get("PMI_RANK", 0)))
    ws = int(os.environ.get("PALS_LOCAL_SIZE", os.environ.get("PMI_SIZE", 1)))
    lr = int(os.environ.get("PALS_LOCAL_RANKID", r % max(ws, 1)))
    return r, ws, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["bare", "xpu", "dist", "sigterm"])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--persistent", type=int, default=1)
    ap.add_argument("--sharing", default="file_system",
                    choices=["file_system", "file_descriptor"])
    args = ap.parse_args()

    rank, world, local = rank_env()
    tag = f"[repro r{rank}]"

    # Match the trainer's preamble exactly (main_dist_aurora.py:39,358) -- a repro
    # that differs from the real process here would be testing a different program.
    os.environ.setdefault("MP_SOCKET_DIR", "/tmp")
    mp.set_sharing_strategy(args.sharing)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"{tag} stage={args.stage} nw={args.num_workers} "
          f"persistent={bool(args.persistent)} sharing={args.sharing} "
          f"world={world} local={local}", flush=True)

    if args.stage == "sigterm":
        # No DataLoader at all. If this throws, the exception is a property of
        # signal-kill on this stack and the nw story collapses.
        print(f"{tag} no loader; raising SIGTERM at t+2s", flush=True)
        time.sleep(2)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(10)
        print(f"{tag} SIGTERM did not terminate us", flush=True)
        return

    device = None
    if args.stage in ("xpu", "dist"):
        # Pin BEFORE init_process_group, per the Aurora rule.
        torch.xpu.set_device(local)
        device = torch.device(f"xpu:{local}")
        print(f"{tag} pinned {device}", flush=True)

    if args.stage == "dist":
        import torch.distributed as dist
        backend = "xccl" if getattr(dist, "is_xccl_available", lambda: False)() else "ccl"
        if backend == "ccl":
            import oneccl_bindings_for_pytorch  # noqa: F401
        # No device_id= on XPU multi-node -- it hangs DataLoader workers, which is
        # precisely the population under test here.
        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=world, rank=rank,
        )
        print(f"{tag} init_process_group({backend}) ok", flush=True)

    loader = DataLoader(
        TinyDataset(),
        batch_size=2,
        num_workers=args.num_workers,
        persistent_workers=bool(args.persistent) and args.num_workers > 0,
        pin_memory=False,
    )

    n = 0
    for batch in loader:
        if device is not None:
            batch = batch.to(device, non_blocking=True)
            # Touch it, so the XPU allocator actually has live blocks at teardown.
            _ = batch.float().mean().item()
        n += 1
        if n >= args.iters:
            break
    print(f"{tag} consumed {n} batches", flush=True)

    if args.stage == "dist":
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
        print(f"{tag} destroy_process_group ok", flush=True)

    # Drop our reference explicitly so the loader destructor runs HERE, inside a
    # region we are printing around, rather than at interpreter shutdown where
    # the ordering is undefined. If the throw lands between these two prints the
    # loader owns it; if it lands after "exiting main" it is shutdown ordering.
    print(f"{tag} releasing loader", flush=True)
    del loader
    import gc
    gc.collect()
    print(f"{tag} loader released cleanly", flush=True)

    print(f"{tag} exiting main (destructors follow)", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
