#!/usr/bin/env python3
"""Generate Aurora asformer-probe eval configs for v1 phase-2 checkpoints.

Mirrors Leo's Polaris configs (configs/heads/sarrarp50/modcon_v1lambda/*) but
retargets:
  - data CSVs   -> flare copy (surg_2_1_v1_probes_aurora/csv)
  - checkpoints -> flare phase2_main_n16g12_weak/e{N}.pth.tar
  - folder      -> flare runs dir
  - encoder wrapper -> vit_encoder_multiclip_v21 (V-JEPA 2.1 dual-modal loader)

One YAML per checkpoint epoch. Config schema identical to the Polaris probe so
numbers are directly comparable (20ep, asformer ctx3, patience 6, 8 classes,
class weights, 3-head multi-LR sweep).
"""
import os

import yaml

CKPT_DIR = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase2_main_n16g12_weak"
CSV_DIR = "/flare/ModCon/ngetty/surg_2_1_v1_probes_aurora/csv"
RUNS_DIR = "/flare/ModCon/ngetty/surg_2_1_v1_probes_aurora/runs/asformer_origsplit"
OUT_DIR = "/lus/flare/projects/ModCon/ngetty/vjepa2/configs/heads/sarrarp50/aurora_v1lambda"

# Validation gate (e29 known Leo F1 75.12, e69 known 69.01) + lambda regime.
EPOCHS = [29, 69, 159, 189, 199, 229, 259]

CLASS_WEIGHTS = [0.250562, 0.662835, 0.169067, 0.171689,
                 0.187031, 3.65359, 1.720042, 1.185184]


def make_cfg(epoch: int) -> dict:
    tag = f"v1_e{epoch}"
    return {
        "cpus_per_task": 16,
        "eval_name": "video_classification_frozen",
        "folder": os.path.join(RUNS_DIR, tag),
        "mem_per_gpu": "220G",
        "nodes": 4,
        "num_workers": 8,
        "resume_checkpoint": True,
        "tag": f"sar-orig-{tag}",
        "tasks_per_node": 4,
        "model_kwargs": {
            "checkpoint": os.path.join(CKPT_DIR, f"e{epoch}.pth.tar"),
            "module_name": "evals.video_classification_frozen.modelcustom.vit_encoder_multiclip_v21",
            "pretrain_kwargs": {
                "encoder": {
                    "checkpoint_key": "target_encoder",
                    "img_temporal_dim_size": 1,
                    "model_name": "vit_large",
                    "patch_size": 16,
                    "tubelet_size": 2,
                    "uniform_power": True,
                    "use_rope": True,
                    "use_sdpa": True,
                },
            },
            "wrapper_kwargs": {
                "max_frames": 128,
                "use_pos_embed": False,
                "preserve_clip_dim": True,
            },
        },
        "experiment": {
            "classifier": {
                "name": "asformer",
                "asformer_kwargs": {
                    "num_clips": 3,
                    "tokens_per_clip": 8,
                    "temporal_tokens": 24,
                    "num_layers": 10,
                    "num_heads": 8,
                    "mlp_ratio": 4.0,
                    "dropout": 0.0,
                    "return_sequence": True,
                },
            },
            "data": {
                "dataset_type": "VideoDataset",
                "dataset_train": os.path.join(CSV_DIR, "sarrarp50_actions_4fps_asformer_ctx3_seq_train.csv"),
                "dataset_val": os.path.join(CSV_DIR, "sarrarp50_actions_4fps_asformer_ctx3_seq_val.csv"),
                "num_classes": 8,
                "frames_per_clip": 16,
                "frame_step": 1,
                "resolution": 384,
                "num_segments": 3,
                "num_views_per_segment": 1,
                "sequence_labels": True,
                "normalization": None,
                "class_weights": CLASS_WEIGHTS,
            },
            "optimization": {
                "batch_size": 4,
                "num_epochs": 20,
                "use_bfloat16": True,
                "loss_smoothing_weight": 0.15,
                "loss_smoothing_threshold": 4.0,
                "use_pos_embed": False,
                "multihead_kwargs": [
                    {"warmup": 0.05, "start_lr": 0.0001, "lr": 0.001,
                     "final_lr": 1.0e-05, "weight_decay": 0.05, "final_weight_decay": 0.4},
                    {"warmup": 0.05, "start_lr": 3.0e-05, "lr": 0.0003,
                     "final_lr": 1.0e-05, "weight_decay": 0.05, "final_weight_decay": 0.4},
                    {"warmup": 0.05, "start_lr": 1.0e-05, "lr": 0.0001,
                     "final_lr": 1.0e-05, "weight_decay": 0.1, "final_weight_decay": 0.4},
                ],
            },
        },
        "early_stop_patience": 6,
        "log_val_f1": True,
        "write_status_json": True,
        "save_every_iters": 200,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for ep in EPOCHS:
        cfg = make_cfg(ep)
        path = os.path.join(OUT_DIR, f"v1_e{ep}.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
