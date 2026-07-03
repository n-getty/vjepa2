import os, time
for _v in ("PALS_LOCAL_RANKID","PMI_LOCAL_RANK","LOCAL_RANK"):
    if _v in os.environ: os.environ["ZE_AFFINITY_MASK"]=os.environ[_v]; break
import copy, torch, torch.distributed as dist, torch.nn as nn
from functools import partial
from src.models.utils.modules import Block
rank=int(os.environ.get("PALS_RANKID","0")); ws=int(os.environ.get("WORLD_SIZE","1")); lws=int(os.environ.get("LOCAL_WORLD_SIZE","12"))
torch.xpu.set_device(0); os.environ.setdefault("MASTER_PORT","29722")
def log(m): print(f"[r{rank} +{time.time()-t0:6.1f}s] {m}", flush=True)
t0=time.time()
dist.init_process_group("xccl", rank=rank, world_size=ws); log(f"pg init ws={ws}")
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
nn_=ws//lws; mesh=init_device_mesh("xpu",(nn_,lws),mesh_dim_names=("replicate","shard"))
mp=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
wp=partial(transformer_auto_wrap_policy,transformer_layer_cls={Block})
def mk(): return nn.Sequential(*[Block(dim=1664,num_heads=26,use_rope=False,use_sdpa=False) for _ in range(4)]).to("xpu:0")
def wrap(mod,rg):
    f=FSDP(mod,auto_wrap_policy=wp,mixed_precision=mp,sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,device_mesh=mesh,use_orig_params=True,limit_all_gathers=True)
    if not rg:
        for p in f.parameters(): p.requires_grad=False
    return f
# target: frozen, deepcopy BEFORE wrap (mirrors trainer)
raw_t=mk(); tgt=wrap(raw_t, rg=False); log("frozen target FSDP-wrapped")
x=torch.randn(2,196,1664,device="xpu:0")
log("--- FROZEN TARGET forward UNDER no_grad (the trainer's forward_target, FIRST call) ---")
with torch.no_grad():
    with torch.amp.autocast(device_type="xpu",dtype=torch.bfloat16):
        h=tgt(x)
torch.xpu.synchronize(); log(f"NO_GRAD FROZEN FORWARD OK h={tuple(h.shape)}  <-- if this hangs, target_encoder is the culprit")
dist.barrier(); log("PASSED nograd-target"); dist.destroy_process_group()
