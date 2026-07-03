import os, time
for _v in ("PALS_LOCAL_RANKID","PMI_LOCAL_RANK","LOCAL_RANK"):
    if _v in os.environ: os.environ["ZE_AFFINITY_MASK"]=os.environ[_v]; break
import torch, torch.distributed as dist, torch.nn as nn
from functools import partial
from src.models.utils.modules import Block
rank=int(os.environ.get("PALS_RANKID","0")); ws=int(os.environ.get("WORLD_SIZE","1")); lws=int(os.environ.get("LOCAL_WORLD_SIZE","12"))
torch.xpu.set_device(0); os.environ.setdefault("MASTER_PORT","29711")
def log(m): print(f"[r{rank} +{time.time()-t0:6.1f}s] {m}", flush=True)
t0=time.time()
dist.init_process_group("xccl", rank=rank, world_size=ws); log(f"pg init ws={ws}")
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
nn_=ws//lws; mesh=init_device_mesh("xpu",(nn_,lws),mesh_dim_names=("replicate","shard")); log("mesh built")
m=nn.Sequential(*[Block(dim=1664,num_heads=26,use_rope=False,use_sdpa=False) for _ in range(4)]).to("xpu:0")
mp=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16)
wp=partial(transformer_auto_wrap_policy,transformer_layer_cls={Block})
f=FSDP(m,auto_wrap_policy=wp,mixed_precision=mp,sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,device_mesh=mesh,use_orig_params=True,limit_all_gathers=True)
log("FSDP wrapped")
x=torch.randn(2,196,1664,device="xpu:0")
log("--- first FORWARD (FSDP lazy-init all-gather here) ---")
with torch.amp.autocast(device_type="xpu",dtype=torch.bfloat16):
    y=f(x)
torch.xpu.synchronize(); log(f"FORWARD OK y={tuple(y.shape)}")
log("--- first BACKWARD (ReduceScatter intra + AllReduce inter) ---")
y.sum().backward(); torch.xpu.synchronize(); log("BACKWARD OK")
log("FSDP FWD+BWD PASSED")
dist.barrier(); dist.destroy_process_group()
