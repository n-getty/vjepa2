"""Single source of truth for the probe RNG seed.

Why this exists: the seed used to live only as `VJEPA_PROBE_SEED`, read once at
`evals/*/eval.py` import time into a module-local that was never referenced
again. Three things followed from that, all of which silently produced
same-seed runs that were reported as multi-seed results:

1. The YAML `seed:` key was read by NOTHING. Configs declaring `seed: 0/1/2`
   (e.g. the GraSP prod_wide set) differed only in folder/tag.
2. Most GraSP launchers never exported `VJEPA_PROBE_SEED`, and
   `run_asformer_probe_aurora.sh` defaults it to `0` -- so every "3-seed"
   GraSP campaign ran three copies of seed 0.
3. `DistributedSampler` was constructed with no `seed=` (torch default 0) and
   `ShardOrderDistributedSampler` with a hardcoded `seed=0`, so even when the
   global seed did vary, DATA ORDER did not.

Resolution order is env -> YAML -> 0. Env wins because it is set per-rank by
the launcher before the process starts, which is the only mechanism that works
under PALS; `set_probe_seed()` lets a config supply the value when the launcher
did not, and refuses to silently override an explicit env var.
"""

import logging
import os

logger = logging.getLogger(__name__)

_ENV_VAR = "VJEPA_PROBE_SEED"
# The segmentation probe (evals/video_segmentation_frozen/eval.py) predates this
# module and reads a DIFFERENT name, so it was invisible to the whole mechanism:
# exporting VJEPA_PROBE_SEED for a seg run did nothing. Accept both, preferring
# the canonical name, rather than leaving one probe family silently unseeded.
_ENV_ALIASES = (_ENV_VAR, "VJEPA_SEED")

_FROM_ENV = any(k in os.environ for k in _ENV_ALIASES)
_SEED = 0
for _k in _ENV_ALIASES:
    if _k in os.environ:
        _SEED = int(os.environ[_k])
        break


def probe_seed():
    """The seed every probe RNG (torch, numpy, samplers) should derive from."""
    return _SEED


def seed_from_env():
    """Whether the seed was set by the launcher rather than defaulted/config."""
    return _FROM_ENV


def set_probe_seed(seed, source="config"):
    """Set the process-wide probe seed and reseed torch/numpy.

    A seed supplied by config does NOT override one supplied by the
    environment: the launcher is authoritative (it is what varies per job in a
    seed sweep), and silently winning over it would reintroduce exactly the
    class of bug this module exists to prevent. Returns the effective seed.
    """
    global _SEED
    if seed is None:
        return _SEED
    seed = int(seed)
    if _FROM_ENV and seed != _SEED:
        logger.warning(
            "%s=%d from environment overrides %s seed=%d (launcher wins).",
            _ENV_VAR, _SEED, source, seed,
        )
        return _SEED
    _SEED = seed
    _reseed(_SEED)
    logger.info("probe seed set to %d (from %s)", _SEED, source)
    return _SEED


def _reseed(seed):
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
