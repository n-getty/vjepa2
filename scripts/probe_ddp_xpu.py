"""Minimal DDP-on-XPU probe to identify why DDP setup segfaults.

Designed to run via:
    mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
        python scripts/probe_ddp_xpu.py [--mode A|B|C]

Modes:
  A: barrier only (no DDP) — confirms init_process_group works
  B: barrier + DDP a tiny model (no device_id arg)
  C: same as B but with device_id passed to init_process_group
  D: same as B but with MPI.COMM_WORLD.Barrier() before init_process_group
  E: same as B but device_ids=[0] passed to DDP
"""
import argparse
import os
import sys

# Pin XPU via local rank BEFORE torch import.
for k in ("PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "MPI_LOCALRANKID",
          "OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"):
    if k in os.environ:
        os.environ["ZE_AFFINITY_MASK"] = os.environ[k]
        break

import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="B")
    args = ap.parse_args()

    # pin xpu device first
    torch.xpu.set_device(0)

    sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/vjepa2")
    from src.utils.distributed import _get_pmi_env

    rank = int(_get_pmi_env("RANK") or "0")
    world = int(_get_pmi_env("SIZE") or "1")
    print(f"[{rank}] start mode={args.mode} world={world} xpu={torch.xpu.is_available()} count={torch.xpu.device_count()}", flush=True)

    # master addr from PBS nodefile
    if not os.environ.get("MASTER_ADDR"):
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            with open(nodefile) as f:
                os.environ["MASTER_ADDR"] = f.readline().strip()
    os.environ.setdefault("MASTER_PORT", "29501")

    if args.mode == "D":
        from mpi4py import MPI
        print(f"[{rank}] pre-init MPI.Barrier", flush=True)
        MPI.COMM_WORLD.Barrier()

    pg_kwargs = dict(backend="xccl", world_size=world, rank=rank)
    if args.mode == "C":
        pg_kwargs["device_id"] = torch.device("xpu:0")

    print(f"[{rank}] init_process_group pg_kwargs={ {k:str(v) for k,v in pg_kwargs.items()} }", flush=True)
    dist.init_process_group(**pg_kwargs)
    print(f"[{rank}] init_process_group OK", flush=True)

    if args.mode == "A":
        # Just exit cleanly
        dist.destroy_process_group()
        return

    # Make tiny model on xpu:0, wrap with DDP
    model = nn.Linear(64, 64).to("xpu:0")
    print(f"[{rank}] model on xpu:0; wrapping DDP", flush=True)
    if args.mode == "E":
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0])
    else:
        ddp = torch.nn.parallel.DistributedDataParallel(model)
    print(f"[{rank}] DDP wrap OK", flush=True)

    x = torch.randn(4, 64, device="xpu:0")
    y = ddp(x).sum()
    print(f"[{rank}] forward OK y={y.item():.3f}", flush=True)
    y.backward()
    print(f"[{rank}] backward OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
