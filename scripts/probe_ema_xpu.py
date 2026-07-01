"""Reproduce + diagnose the EMA target update bug on XPU + DDP.

Runs 4 variants of the EMA update on the same parameters and reports
whether the target actually moved:

  V1: train.py exact code (torch._foreach_mul_, torch._foreach_add_)
  V2: same as V1 but on .data tensors (bypasses param wrappers)
  V3: explicit Python loop with .mul_/.add_
  V4: same as V3 but on .data tensors

If V1 fails to update target but V2/V3/V4 succeed → bug is in foreach ops
on DDP-wrapped params on XPU.

Run via:
  mpiexec --pmi=pmix -n 12 -ppn 12 --cpu-bind depth --depth 16 \
      python scripts/probe_ema_xpu.py [V1|V2|V3|V4]
"""
import argparse
import os
import sys

# pin XPU before torch import
for k in ("PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "MPI_LOCALRANKID",
          "OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"):
    if k in os.environ:
        os.environ["ZE_AFFINITY_MASK"] = os.environ[k]
        break

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/vjepa2")
from src.utils.distributed import _get_pmi_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["V1", "V2", "V3", "V4", "noddp"], default="V1", nargs="?")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--m", type=float, default=0.99925)
    args = ap.parse_args()

    rank = int(_get_pmi_env("RANK") or "0")
    world = int(_get_pmi_env("SIZE") or "1")
    torch.xpu.set_device(0)

    if not os.environ.get("MASTER_ADDR"):
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            os.environ["MASTER_ADDR"] = open(nodefile).readline().strip()
    os.environ.setdefault("MASTER_PORT", "29504")

    if world > 1:
        dist.init_process_group(backend="xccl", world_size=world, rank=rank)
        if rank == 0:
            print(f"[r0] init dist OK, world={world}", flush=True)

    # Build two identical small models
    torch.manual_seed(42)
    encoder = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 64),
    ).to("xpu:0")
    target = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 64),
    ).to("xpu:0")
    # Match init exactly
    target.load_state_dict(encoder.state_dict())

    # Wrap in DDP BEFORE requires_grad=False (matches trainer order)
    if args.variant != "noddp" and world > 1:
        encoder = DistributedDataParallel(encoder)
        target = DistributedDataParallel(target)
    for p in target.parameters():
        p.requires_grad = False

    # Snapshot initial target
    target_init = [p.detach().clone() for p in target.parameters()]

    # Fake "training": just perturb encoder weights manually each step
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-2)

    m = args.m
    for step in range(args.steps):
        # Fake forward+backward to move encoder
        x = torch.randn(8, 64, device="xpu:0")
        y = encoder(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad()

        # EMA update (variant under test)
        with torch.no_grad():
            if args.variant == "V1":
                # Exact code from train.py
                params_k = []; params_q = []
                for pq, pk in zip(encoder.parameters(), target.parameters()):
                    params_k.append(pk); params_q.append(pq)
                torch._foreach_mul_(params_k, m)
                torch._foreach_add_(params_k, params_q, alpha=1 - m)
            elif args.variant == "V2":
                # V1 with .data
                params_k = []; params_q = []
                for pq, pk in zip(encoder.parameters(), target.parameters()):
                    params_k.append(pk.data); params_q.append(pq.data)
                torch._foreach_mul_(params_k, m)
                torch._foreach_add_(params_k, params_q, alpha=1 - m)
            elif args.variant == "V3":
                # Explicit Python loop
                for pq, pk in zip(encoder.parameters(), target.parameters()):
                    pk.mul_(m).add_(pq, alpha=1 - m)
            elif args.variant == "V4":
                # Explicit loop with .data
                for pq, pk in zip(encoder.parameters(), target.parameters()):
                    pk.data.mul_(m).add_(pq.data, alpha=1 - m)
            elif args.variant == "noddp":
                # V1 without DDP wrap
                params_k = []; params_q = []
                for pq, pk in zip(encoder.parameters(), target.parameters()):
                    params_k.append(pk); params_q.append(pq)
                torch._foreach_mul_(params_k, m)
                torch._foreach_add_(params_k, params_q, alpha=1 - m)

    # Measure: did target move at all?
    target_now = [p.detach().clone() for p in target.parameters()]
    encoder_now = [p.detach().clone() for p in encoder.parameters()]

    if rank == 0:
        print(f"\n=== variant {args.variant}: {args.steps} EMA steps with m={m} ===", flush=True)
        for i, (init, now, enc) in enumerate(zip(target_init, target_now, encoder_now)):
            tgt_drift = (now - init).norm().item()
            tgt_norm = init.norm().item()
            tgt_rel = tgt_drift / tgt_norm * 100 if tgt_norm > 0 else 0
            enc_drift = (enc - init).norm().item()
            enc_rel = enc_drift / tgt_norm * 100 if tgt_norm > 0 else 0
            print(f"  param[{i}] shape={tuple(init.shape)}: "
                  f"||tgt-init||/||init||={tgt_rel:.4f}%  "
                  f"||enc-init||/||init||={enc_rel:.4f}%  "
                  f"ratio tgt/enc={tgt_rel/enc_rel:.4f}" if enc_rel > 0 else "")
        # Expected: tgt/enc ratio approaches (1 - m^N) / (N * (1-m)) for linear drift
        # For N=100, m=0.99925: m^100 = 0.928, so window = 100*(1-m) * sum/N ≈ ?
        # Crude estimate: after 100 steps target should have moved roughly 50% of encoder drift
        print(f"\nExpected for {args.steps} steps, m={m}: target should be at ~30-70% of encoder's drift")
        print(f"If target drift is 0% — EMA op was a no-op")

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
