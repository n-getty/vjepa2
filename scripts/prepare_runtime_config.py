#!/usr/bin/env python3

import argparse
from pathlib import Path

import yaml


ALLOWED_GPUS_PER_NODE = {2, 4, 6, 8, 12, 24}
DEFAULT_BASE_GPU_COUNT = 4  # historical Polaris baseline; overridden by YAML tasks_per_node


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a runtime YAML adapted to a target (nodes x gpus-per-node) topology. "
            "Default is strong scaling: fix the global batch size; per-rank batch "
            "shrinks as GPUs grow. Pass --weak-scale to keep per-rank batch fixed "
            "and let the global batch grow with GPU count (throughput mode)."
        )
    )
    parser.add_argument("config", help="Path to the base YAML config")
    parser.add_argument("--root", required=True, help="Repository root")
    parser.add_argument("--num-gpus", required=True, type=int,
                        choices=sorted(ALLOWED_GPUS_PER_NODE),
                        help="GPUs per node (e.g. 4 on Polaris, 12 on Aurora).")
    parser.add_argument("--num-nodes", type=int, default=1,
                        help="Number of nodes (default 1).")
    parser.add_argument("--weak-scale", action="store_true",
                        help="Weak scaling: keep per-rank batch fixed at the base "
                             "value; global batch grows linearly with total GPU "
                             "count. Default is strong scaling (preserve global "
                             "batch by shrinking per-rank batch).")
    # Backwards-compat alias: the polaris-multinode branch shipped with the
    # terms inverted (called weak scaling "--strong-scale"). Accept the old
    # name silently so existing scripts keep working.
    parser.add_argument("--strong-scale", dest="weak_scale", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--folder-base", default=None,
                        help="If set, replaces the YAML's folder with "
                             "<folder-base>/<basename(folder)>; the topology suffix "
                             "(_g{N} / _n{N}g{G} / _weak) is still appended.")
    return parser.parse_args()


def topology_suffix(num_gpus: int, num_nodes: int, base_gpu_count: int) -> str:
    if num_nodes == 1 and num_gpus == base_gpu_count:
        return ""
    if num_nodes == 1:
        return f"_g{num_gpus}"
    return f"_n{num_nodes}g{num_gpus}"


def scale_local_batch(base_batch: int, total_gpus: int, weak_scale: bool,
                      base_gpu_count: int) -> int:
    """Return per-rank batch under strong or weak scaling.

    Weak scaling: per-rank batch fixed at the base value; global batch grows.
    Strong scaling (default): global batch fixed at base x base_gpu_count;
    per-rank batch shrinks as GPUs grow.
    """
    if weak_scale:
        return base_batch
    scaled = base_batch * base_gpu_count
    if scaled % total_gpus != 0:
        raise ValueError(
            f"Cannot preserve global batch from base batch_size={base_batch} "
            f"with total_gpus={total_gpus}"
        )
    new_batch = scaled // total_gpus
    if new_batch < 1:
        raise ValueError(
            f"Scaled local batch would be invalid: base batch_size={base_batch}, "
            f"total_gpus={total_gpus}; use --weak-scale or pick fewer GPUs."
        )
    return new_batch


def remap_folder(folder: str, num_gpus: int, num_nodes: int, base_gpu_count: int) -> str:
    suffix = topology_suffix(num_gpus, num_nodes, base_gpu_count)
    return f"{folder}{suffix}" if suffix else folder


def remap_checkpoint_path(path_value, num_gpus, num_nodes, base_gpu_count):
    suffix = topology_suffix(num_gpus, num_nodes, base_gpu_count)
    if not path_value or not suffix:
        return path_value
    path = Path(path_value)
    return str(path.parent.with_name(f"{path.parent.name}{suffix}") / path.name)


def remap_repo_checkpoint_path(path_value, num_gpus, num_nodes, repo_root, base_gpu_count):
    suffix = topology_suffix(num_gpus, num_nodes, base_gpu_count)
    if not path_value or not suffix:
        return path_value

    raw_path = Path(path_value)
    try:
        resolved_path = raw_path.resolve()
    except OSError:
        return path_value

    checkpoints_root = (repo_root / "checkpoints").resolve()
    try:
        resolved_path.relative_to(checkpoints_root)
    except ValueError:
        return path_value

    return remap_checkpoint_path(str(raw_path), num_gpus, num_nodes, base_gpu_count)


def main():
    args = parse_args()

    repo_root = Path(args.root).resolve()
    config_path = Path(args.config).resolve()
    total_gpus = args.num_gpus * args.num_nodes

    runtime_dir = (
        f"g{args.num_gpus}" if args.num_nodes == 1
        else f"n{args.num_nodes}g{args.num_gpus}"
    )
    if args.weak_scale:
        runtime_dir += "_weak"
    runtime_root = repo_root / ".runtime_configs" / runtime_dir

    with config_path.open("r") as handle:
        cfg = yaml.safe_load(handle)

    # The "base" topology is what the YAML was authored for; we derive it from
    # the YAML's tasks_per_node so this script works for both Polaris (4 GPUs)
    # and Aurora (12 tiles) without per-platform branching.
    base_gpu_count = int(cfg.get("tasks_per_node", DEFAULT_BASE_GPU_COUNT))

    cfg["nodes"] = args.num_nodes
    cfg["tasks_per_node"] = args.num_gpus
    base_folder = cfg["folder"]
    if args.folder_base:
        base_folder = str(Path(args.folder_base) / Path(base_folder.rstrip("/")).name)
    cfg["folder"] = remap_folder(base_folder, args.num_gpus, args.num_nodes, base_gpu_count)
    if args.weak_scale and "_weak" not in cfg["folder"]:
        cfg["folder"] = cfg["folder"].rstrip("/") + "_weak"

    data_cfg = cfg.get("data", {})
    base_batch = int(data_cfg["batch_size"])
    data_cfg["batch_size"] = scale_local_batch(
        base_batch, total_gpus, args.weak_scale, base_gpu_count
    )
    cfg["data"] = data_cfg

    meta_cfg = cfg.get("meta", {})
    meta_cfg["pretrain_checkpoint"] = remap_repo_checkpoint_path(
        meta_cfg.get("pretrain_checkpoint"), args.num_gpus, args.num_nodes, repo_root, base_gpu_count
    )
    meta_cfg["read_checkpoint"] = remap_repo_checkpoint_path(
        meta_cfg.get("read_checkpoint"), args.num_gpus, args.num_nodes, repo_root, base_gpu_count
    )
    cfg["meta"] = meta_cfg

    optimization_cfg = cfg.get("optimization", {})
    optimization_cfg["anneal_ckpt"] = remap_repo_checkpoint_path(
        optimization_cfg.get("anneal_ckpt"), args.num_gpus, args.num_nodes, repo_root, base_gpu_count
    )
    cfg["optimization"] = optimization_cfg

    relative_cfg_path = config_path.relative_to(repo_root)
    output_path = runtime_root / relative_cfg_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    print(output_path)


if __name__ == "__main__":
    main()
