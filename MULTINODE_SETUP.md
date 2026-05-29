# V-JEPA 2.1 Multi-Node Training on Polaris

This document covers a multi-node Polaris (PBS + PALS/MPICH) launch path for the
V-JEPA 2 and 2.1 training code in this repo. Upstream's only distributed
launcher (`app/main_distributed.py`) uses SLURM/submitit and does not work on
Polaris.

## Quick start

```bash
# 1-node baseline (4 GPUs, Polaris)
VJEPA_NUM_NODES=1 VJEPA_NUM_GPUS=4 VJEPA_STRONG_SCALE=1 \
  VJEPA_FOLDER_BASE=/eagle/<project>/<you>/checkpoints/vjepa21 \
  VJEPA_ACCOUNT=<your-PBS-allocation> VJEPA_PARTITION=debug VJEPA_TIME_MIN=60 \
  VJEPA_PYTHON=/path/to/your/venv/bin/python \
  ./run_three_phases_polaris.sh \
    configs/train_2_1/vitl16/pretrain-256px-16f.yaml

# 2-node (8 GPUs, debug-scaling or preemptable)
VJEPA_NUM_NODES=2 VJEPA_NUM_GPUS=4 VJEPA_STRONG_SCALE=1 \
  VJEPA_FOLDER_BASE=... VJEPA_ACCOUNT=... VJEPA_PYTHON=... \
  VJEPA_PARTITION=debug-scaling VJEPA_TIME_MIN=60 \
  ./run_three_phases_polaris.sh <yaml>

# 4-node strong-scaling
VJEPA_NUM_NODES=4 VJEPA_NUM_GPUS=4 VJEPA_STRONG_SCALE=1 \
  VJEPA_FOLDER_BASE=... VJEPA_ACCOUNT=... VJEPA_PYTHON=... \
  VJEPA_PARTITION=preemptable VJEPA_TIME_MIN=60 \
  ./run_three_phases_polaris.sh <yaml>
```

`run_three_phases_polaris.sh` chains all YAMLs you pass via PBS `afterok`
dependencies. Each phase becomes a separate PBS job; subsequent phases run
after the previous succeeds.

## Env-var reference

| Env var | Default | Purpose |
|---|---|---|
| `VJEPA_NUM_NODES` | required | PBS `-l select=N` |
| `VJEPA_NUM_GPUS` | `4` | GPUs per node (2/4/8 supported; 4 is standard on Polaris) |
| `VJEPA_STRONG_SCALE` | `0` (weak) | `1` keeps per-rank batch fixed; `0` preserves global batch by shrinking per-rank |
| `VJEPA_PARTITION` | `prod` | `debug` (≤2n,1h), `debug-scaling` (≤10n,1h), `preemptable` (≤10n,72h), `prod` (≥10n,24h) |
| `VJEPA_ACCOUNT` | required | PBS `-A` |
| `VJEPA_PYTHON` | required | Path to the `python` inside the venv used inside PBS (also used by the prep tool on the login node) |
| `VJEPA_TIME_MIN` | `720` | Walltime in minutes |
| `VJEPA_FILESYSTEMS` | `home:eagle` | PBS `-l filesystems=` |
| `VJEPA_FOLDER_BASE` | unset | If set, replaces the YAML's `folder:` with `<base>/<basename(folder)>` |
| `VJEPA_STAGE_DATA` | `0` | `1` to rsync `data.datasets` to `/local/scratch` per node. Works with `dataset_type: WebDataset` configs (each entry is a shard directory). Does NOT work for `dataset_type: VideoDataset` configs that list CSVs — the CSVs would get staged but the absolute video paths inside them still point at `/eagle`. |
| `VJEPA_DISABLE_AWS_OFI` | `0` | `1` to disable the AWS OFI NCCL plugin (only set if you see NCCL hangs) |
| `VJEPA_DRY_RUN` | `0` | `1` builds PBS scripts but doesn't qsub |

## Architecture

```
run_three_phases_polaris.sh         <- wrapper, env-var-driven
  └─ scripts/prepare_runtime_config.py  <- rewrites YAML per topology
  └─ app/main_dist_polaris.py           <- login submits PBS / compute trains
       └─ src/utils/distributed.py       <- init_distributed() with PMI fallback
       └─ app/scaffold.py + app/vjepa_2_1/train.py  <- unchanged trainer
```

**`scripts/prepare_runtime_config.py`** generates `.runtime_configs/n{N}g{G}[_strong]/...`
copies of the source YAML with `nodes`, `tasks_per_node`, and per-rank
`batch_size` adjusted. Folder gets a `_n{N}g{G}` suffix to keep runs isolated.

**`app/main_dist_polaris.py`** has two modes:
- **Login**: reads YAML, snapshots code into `<folder>/code/`, writes a PBS
  script with the Polaris env, qsubs it. Supports `--depends_on <jobid>` for
  afterok chaining and `--dry_run` for inspection. Requires `--account` (or
  `$VJEPA_ACCOUNT`) and `--venv` (or `$PYTHON_BIN_VENV`).
- **Compute** (`--train_mode`): pins GPU from `PMI_LOCAL_RANK`, sets
  `MASTER_ADDR` from `$PBS_NODEFILE` head, calls `init_distributed()` (which
  picks up PMI), then hands off to `app.scaffold.main`.

**`src/utils/distributed.py`** `init_distributed()` resolves rank/world-size
via this fallback chain: explicit `(rank, world_size)` arg → SLURM env →
PMI/PALS/OMPI env (Polaris) → torchrun-style `RANK/WORLD_SIZE` → single-process
`(1, 0)`. Don't add new scheduler branches; extend `_PMI_VAR_CHAINS` instead.

## Measured scaling on Polaris (ViT-L, 256px, bs=4/rank, fps=4)

Date: 2026-05-20. All numbers in steady-state median (first 50 iters dropped).

| Topology | NCCL | iter ms | bwd ms | sps/GPU | sps total | per-GPU scaling vs 1n |
|---|---|---:|---:|---:|---:|---:|
| 1n × 4 GPU (NVLink) | stock | 1186 | 577 | 3.37 | 13.5 | 1.00× |
| 2n × 4 GPU | stock | 1623 | 1014 | 2.46 | 19.7 | 0.73× |
| **2n × 4 GPU** | **AWS OFI** | **1230** | **609** | **3.25** | **26.0** | **0.96×** |
| 4n × 4 GPU | stock | 1679 | ~1050 | 2.38 | 38.1 | 0.71× |
| **4n × 4 GPU** | **AWS OFI** | **1193** | **601** | **3.35** | **53.6** | **0.99×** |

The AWS OFI plugin recovers the inter-node penalty almost entirely. The bottleneck
without it is NCCL allreduce latency in `backward()` — confirmed via phase-level
CUDA-event timing (`src/utils/logging.py:PhaseTimer`, CSV columns
`fwd-target-ms`, `fwd-context-ms`, `backward-ms`, `opt-step-ms`, `ema-ms`).

## WebDataset (for `--stage_data`)

The `dataset_type: WebDataset` path lets you read video pretraining data from
`.tar` shards instead of CSV-of-video-paths. It's the only `data.datasets`
shape that `--stage_data` / `VJEPA_STAGE_DATA=1` can actually rsync to
`/local/scratch` and rebind to via `--local_data_root`.

Build shards from CSV manifests with `app/create_webdataset.py` (each input
CSV becomes its own subdirectory of `.tar` shards plus a `metadata.json`):

```bash
python -m app.create_webdataset \
    --csvs path_<dataset_a>.csv path_<dataset_b>.csv ... \
    --output_dir /eagle/<project>/<you>/data/webdataset \
    --samples_per_shard 1000
```

The resulting layout is what `dataset_type: WebDataset` expects (each input
CSV becomes a subdirectory named after its stem):

```
/eagle/<project>/<you>/data/webdataset/
  <dataset_a>/
    <dataset_a>-000000.tar
    <dataset_a>-000001.tar
    ...
    metadata.json
  <dataset_b>/
    ...
```

In the YAML, point `data.datasets` at the per-dataset directories and set
`dataset_type: WebDataset`. Then either run as-is, or pass
`VJEPA_STAGE_DATA=1` to copy them to `/local/scratch` per node before training.

Sanity-check shards with `app/check_webdataset.py` before a long run.

## Hardware note

Polaris compute nodes have 4 A100/node + Slingshot inter-node. An 8-GPU run
therefore requires 2 Polaris nodes and pays inter-node allreduce cost. Configs
shaped `nodes:1 tasks_per_node:8` were authored for a different platform
(DGX A100, 8 GPU/node, NVLink intra-node) and will be rewritten by
`prepare_runtime_config.py` to fit 4 GPU/node × N nodes.

## Gotchas

1. **bs=1/rank crashes** in `app/vjepa_2_1/train.py` (`d_ij.unsqueeze(2)`
   IndexError) — the loss `d_weights` path assumes batch_size ≥ 2. Avoid
   topologies that would force per-rank bs=1 (e.g. 4 nodes × 4 GPUs with base
   global batch 16 and weak-scaling). Use strong-scaling instead.
2. **AWS OFI plugin is on by default for multi-node**, auto-disabled for
   single-node (it segfaults at DDP init when there's no inter-node traffic).
   If you ever see NCCL hangs on multi-node (rare), `VJEPA_DISABLE_AWS_OFI=1`
   falls back to the slower stock NCCL.
3. **Predictor needs `find_unused_parameters=True`** — confirmed via ablation;
   removing it crashes with "Expected to have finished reduction in the prior
   iteration". This is a known DDP overhead but cannot be turned off without
   refactoring the predictor.
4. **`broadcast_buffers=False`** on DDP wraps had no measurable effect.
5. **Polaris reverse-GPU-affinity script** had no measurable effect once AWS
   OFI was enabled — AWS OFI plugin handles NIC binding internally.
6. **Login-node Python**: the venv `python` you pass via `$VJEPA_PYTHON` may
   be a compute-node-only conda module (it will fail to import torch on login).
   For login-node tooling (`prepare_runtime_config.py`, dry-runs) use a
   lightweight Python that has `pyyaml` and `torch` available on the login
   filesystem.

## Inspecting a run

```bash
# Compute median per-phase timings from a run
./scripts/analyze_phase_csv.py /eagle/<project>/<you>/checkpoints/vjepa21/phase1_warmup_n2g4

# Compare multiple runs (e.g. for ablation deltas)
./scripts/analyze_phase_csv.py \
    .../phase1_warmup .../phase1_warmup_n2g4_strong
```

CSV columns (per rank, in `<folder>/log_r{rank}.csv`):
`epoch, itr, loss, iter-time(ms), gpu-time(ms), dataload-time(ms),`
`fwd-target-ms, fwd-context-ms, backward-ms, opt-step-ms, ema-ms`

The last 5 columns are written by `src/utils/logging.py:PhaseTimer`.
