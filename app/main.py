# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import multiprocessing as mp
import pprint
from pathlib import Path

import yaml

from app.scaffold import main as app_main
from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument(
    "--device_type",
    type=str,
    default="cuda",
    choices=['cuda', 'xpu'],
    help="device to use for training",
)
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    help="which devices to use on local machine (e.g., cuda:0, xpu:1)",
)
parser.add_argument(
    "--debugmode",
    action="store_true",
    help="Setting this to true will not spin up new processes. "
    "The main code runs the main process, which makes it easier to \
    debug with checkpointing.",
)


def process_main(rank, fname, world_size, devices, device_type):
    import os

    # This is handled by the training script now
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")

    # Add device info to params
    params['device'] = devices[rank]
    params['device_type'] = device_type

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = params["folder"]
        params_path = os.path.join(folder, "params-pretrain.yaml")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    dist_backend = 'nccl' if device_type == 'cuda' else 'ccl'
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size), backend=dist_backend)
    logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch the app with loaded config
    app_main(params["app"], args=params)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.devices is None:
        if args.device_type == "cuda":
            args.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        elif args.device_type == "xpu":
            # Assuming torch xpu has similar interface
            # This part might need adjustment based on intel_extension_for_pytorch
            try:
                import intel_extension_for_pytorch as ipex
                args.devices = [f"xpu:{i}" for i in range(ipex.xpu.device_count())]
            except ImportError:
                args.devices = []

    if args.debugmode:
        process_main(rank=0, fname=args.fname, world_size=1, devices=args.devices, device_type=args.device_type)
    else:
        num_devices = len(args.devices)
        mp.set_start_method("spawn")
        for rank in range(num_devices):
            mp.Process(target=process_main, args=(rank, args.fname, num_devices, args.devices, args.device_type)).start()
