# Handoff: continuing the v2 LR-warmup phase-2 run on debug-scaling

**Audience:** Leo (or anyone with their own Aurora `debug-scaling` slot).
**Goal:** advance the **v2 LR-warmup** pretraining run identically to how we run
it, using *your own* `debug-scaling` allocation, so both v1 and v2 progress in
parallel (Aurora `debug-scaling` is `max_run=1` **per user**, so two users = two
concurrent chains).

---

## TL;DR

**One-time setup** (the chain uses `module load frameworks`, NOT a venv — but a
few deps are not in the base frameworks image and must be installed into *your
own* per-user site, see "Environment" below):

```bash
module load frameworks
cd /lus/flare/projects/ModCon/ngetty/vjepa2
pip install --user -e .          # installs webdataset, decord, timm, iopath, braceexpand
```

**Then launch the chain:**

```bash
qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/phase2_chain_debugscaling_lrwarmup.sh
```

That's it. The script self-resubmits (`afterany`) before each training slice, so
one `qsub` launches a self-perpetuating 1-hour-slice chain that runs until the
run hits 600 epochs. Check progress any time with `qstat -u $USER`.

> **Account note:** the script has `#PBS -A AuroraGPT`. If you must charge a
> different project, edit that line (or `qsub -A <your_project> ...`). Everything
> else works unmodified — the checkpoint dir is group-readable under ModCon.

---

## What this run is

`phase2_main_n16g12_weak_lrwarmup` — identical to the v1 production run
(`phase2_main_n16g12_weak`) in **every** respect except the optimizer LR schedule:

| | v1 (production) | v2 (this run) |
|---|---|---|
| `start_lr` | 5.25e-4 | **5.0e-5** |
| `warmup` (epochs) | 0 | **30** |
| `lr` / `final_lr` | 5.25e-4 | 5.25e-4 |
| everything else | — | identical |

So v2 ramps LR 5e-5 → 5.25e-4 over the first 30 epochs, then holds flat — testing
whether easing the phase1→phase2 LR step-shock changes training/downstream
behavior. `lambda_progressive` (context-loss weight ramp, iter 15k–30k =
**epoch 150–300**), EMA (0.99925), data, masks, and model are all the same as v1.

**Current state:** epoch **34 / 600** (as of 2026-06-17). The interesting regime
is **epoch 150+**, where the λ ramp turns on — this run needs to get there.

---

## Key paths

| What | Path |
|---|---|
| Repo root | `/lus/flare/projects/ModCon/ngetty/vjepa2` |
| Chain script | `scripts/phase2_chain_debugscaling_lrwarmup.sh` |
| Checkpoint dir | `/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak_lrwarmup` |
| Run config | `<ckpt_dir>/params-pretrain.yaml` (already staged; do not regenerate) |
| Logs | `/flare/ModCon/ngetty/logs/phase2_ds_lrwarmup.o<jobid>` |
| Resume source | `<ckpt_dir>/latest.pth.tar` (auto-loaded; `load_checkpoint: true`) |
| Epoch snapshots | `<ckpt_dir>/e{N}.pth.tar` every 10 epochs (`save_every_freq: 10`) |

---

## Environment — frameworks + per-user pip (NOT a venv)

There is **no conda env or venv** for this project. The chain script runs
`module load frameworks` and then calls the system frameworks Python directly:
`/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python`.

**However, `module load frameworks` alone is NOT sufficient.** The frameworks
image isolates a *per-user* site-packages dir
(`~/.local/aurora/frameworks/2025.3.1/lib/python3.12/site-packages`, auto-added
to `sys.path` by the module). Several of this project's deps are **not** in the
base image and live only in the user-site — so each user must install them once
into *their own* user-site:

| dep | in base frameworks image? |
|---|---|
| torch, numpy, einops, yaml (PyYAML), PIL (pillow) | ✅ yes |
| **webdataset, decord, timm, iopath, braceexpand** | ❌ **no — must `pip install --user`** |

The repo's `requirements.txt` lists exactly these, so the one-time setup is just:

```bash
module load frameworks
cd /lus/flare/projects/ModCon/ngetty/vjepa2
pip install --user -e .
```

Verify before submitting (run as yourself, with frameworks loaded):

```bash
python -c "import torch, webdataset, decord, timm, iopath, braceexpand, einops, yaml; print('deps OK')"
```

Notes:
- The repo itself does **not** need to be installed to be importable — the chain
  script `cd`s into the repo root before launching, so `src`/`app` resolve from
  cwd. `pip install -e .` is only to pull the missing third-party deps (the `-e`
  keeps the repo editable/in-place). If you prefer, `pip install --user -r
  requirements.txt` does the same without touching the repo.
- Reference pinned versions in our working user-site: webdataset 1.0.2,
  decord 0.6.0, braceexpand 0.1.7, timm 1.0.x.
- Don't create a conda env — the Intel-tuned torch/IPEX/oneCCL only comes from
  `module load frameworks`.

---

## How the chain works (so you can reason about it)

Each 1-hour slice does, in order:

1. **Reads `latest.pth.tar` epoch.** If `>= 600`, the chain stops cleanly.
2. **Self-resubmits** with `-W depend=afterany:$PBS_JOBID` *before* training — so
   the chain survives both a clean walltime exit (`exit 0`) and a crash. It only
   resubmits if no `debug-scaling` job of yours is already `Q` (avoids piling up).
3. **Lock guard** (`<ckpt_dir>/.training.lock`): if another job is *already
   training this exact run* (holds the lock and is `R`), this slice skips training
   but keeps the chain alive. Prevents two jobs corrupting `latest.pth.tar`. The
   lock is released on any exit (`trap ... EXIT`) and a stale lock (holder not
   `R`) is taken over automatically. **You generally won't hit this** unless you
   and we both point a chain at the *same* v2 dir — don't do that; you run v2,
   we run v1 (different dirs, different locks).
4. **Per-node /tmp staging** (`scripts/stage_node_shards.py`): each of the 16
   nodes copies only the disjoint shard slice its 12 local ranks will read onto
   local `/tmp`, then training reads from there (`WDS_LOCAL_SLICING=1`,
   `--local_data_root`). This is what avoids the flare I/O contention that caused
   50–100 s iter spikes. ~30 s to stage, ~41 GiB/node.
5. **Training:** `mpiexec -n 192 -ppn 12` → `app.main_dist_aurora --train_mode`.

`ipe=100`, so ~1 slice ≈ several epochs depending on backfill. Resume is exact:
optimizer/scaler/EMA/scheduler all restored, so slicing has no effect on the
trajectory vs. one long run.

---

## Monitoring

```bash
qstat -u $USER -w                       # R = training, H = held resubmit, Q = queued
# current epoch:
python -c "import torch; print(torch.load('/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak_lrwarmup/latest.pth.tar', map_location='cpu', weights_only=False)['epoch'])"
# latest slice log:
ls -t /flare/ModCon/ngetty/logs/phase2_ds_lrwarmup.o* | head -1 | xargs tail -30
```

Healthy slice log shows: `progress: epoch N / 600` → `Chained next job: ...` →
`--- staging complete ---` → periodic `log_stats ... loss:` lines.

---

## Stopping the chain

The chain stops itself at epoch 600. To stop early:

```bash
# delete the running + held jobs (the held one is the next link)
qstat -u $USER | grep phase2_ds_lrwarmup | awk '{print $1}' | xargs qdel
rm -f /flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak_lrwarmup/.training.lock
```

Killing only the running job still leaves the `afterany`-held successor, which
will start the next slice — delete both to fully stop.

---

## Gotchas (learned the hard way)

- **One chain per ckpt dir.** Two simultaneous chains on the same dir race on
  `latest.pth.tar`. The lock guard mitigates but don't rely on it across users —
  you take v2, we take v1.
- **`debug-scaling` is `max_run=1` per user, 1 h walltime, ≤256 nodes.** That's
  why splitting v1/v2 across two people's slots is the win.
- **Don't regenerate `params-pretrain.yaml`.** It's already staged with the
  warmup schedule. The script's pre-flight regen block only fires if the file is
  missing, and it would emit a *v1*-style config — so just leave the existing
  file in place.
- **Environment: frameworks + per-user pip, NO venv/conda.** See the dedicated
  section below — `module load frameworks` alone is *not* enough; you need the
  one-time `pip install --user -e .`.
- **CSV-of-video configs can't be staged, but this run uses WebDataset shards**,
  which is exactly what `stage_node_shards.py` handles.
