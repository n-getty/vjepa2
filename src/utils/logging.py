# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import subprocess
import sys

import torch


def gpu_timer(closure, log_timings=True):
    """Helper to time gpu-time to execute closure(). Works on CUDA or XPU."""
    if torch.cuda.is_available():
        _events = torch.cuda
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        _events = torch.xpu
    else:
        _events = None
    log_timings = log_timings and _events is not None

    elapsed_time = -1.0
    if log_timings:
        start = _events.Event(enable_timing=True)
        end = _events.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        _events.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


class PhaseTimer:
    """Record device-event-based elapsed-times around named phases of train_step.

    Cheaper than torch.cuda.synchronize() at each phase boundary because we
    enqueue events on the default stream and only sync once at the very end
    (the outer gpu_timer call already does that). Returns a dict[name -> ms].

    Works on CUDA and XPU. Silently disabled when neither accelerator is present.
    """

    def __init__(self, enabled=True):
        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self._device = "xpu"
        else:
            self._device = None
        self.enabled = enabled and self._device is not None
        self.events = []
        self.names = []

    def _new_event(self):
        if self._device == "cuda":
            return torch.cuda.Event(enable_timing=True)
        return torch.xpu.Event(enable_timing=True)

    def _sync(self):
        if self._device == "cuda":
            torch.cuda.synchronize()
        else:
            torch.xpu.synchronize()

    def mark(self, name):
        if not self.enabled:
            return
        ev = self._new_event()
        ev.record()
        self.events.append(ev)
        self.names.append(name)

    def to_dict(self):
        if not self.enabled or len(self.events) < 2:
            return {}
        self._sync()
        out = {}
        for i in range(len(self.events) - 1):
            label = f"{self.names[i]}->{self.names[i+1]}"
            out[label] = self.events[i].elapsed_time(self.events[i + 1])
        return out


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class CSVLogger(object):

    def __init__(self, fname, *argv, **kwargs):
        self.fname = fname
        self.types = []
        mode = kwargs.get("mode", "+a")
        self.delim = kwargs.get("delim", ",")
        # -- print headers
        with open(self.fname, mode) as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=self.delim, file=f)
                else:
                    print(v[1], end="\n", file=f)

    def log(self, *argv):
        with open(self.fname, "+a") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = self.delim if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jepa_rootpath():
    this_file = os.path.abspath(__file__)
    return "/".join(this_file.split("/")[:-3])


def git_information():
    jepa_root = jepa_rootpath()
    try:
        resp = (
            subprocess.check_output(["git", "-C", jepa_root, "rev-parse", "HEAD", "--abbrev-ref", "HEAD"])
            .decode("ascii")
            .strip()
        )
        commit, branch = resp.split("\n")
        return f"branch: {branch}\ncommit: {commit}\n"
    except Exception:
        return "unknown"
