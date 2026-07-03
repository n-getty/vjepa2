import os, time
for _v in ("PALS_LOCAL_RANKID","PMI_LOCAL_RANK","LOCAL_RANK"):
    if _v in os.environ: os.environ["ZE_AFFINITY_MASK"]=os.environ[_v]; break
import torch, torch.distributed as dist
rank=int(os.environ.get("PALS_RANKID","0"))
ws=int(os.environ.get("WORLD_SIZE") or os.environ.get("PMIX_SIZE") or 1)
lws=int(os.environ.get("LOCAL_WORLD_SIZE","12"))
torch.xpu.set_device(0)
os.environ.setdefault("MASTER_PORT","29700")
def log(m): print(f"[r{rank} +{time.time()-t0:6.1f}s] {m}", flush=True)
t0=time.time()
dist.init_process_group("xccl", rank=rank, world_size=ws); log(f"pg init ws={ws} lws={lws}")
# bare default-group all_reduce (baseline: does ANY xccl collective work at this scale?)
x=torch.ones(1024,1024,device="xpu:0")
dist.all_reduce(x); torch.xpu.synchronize(); log(f"default all_reduce OK sum={x.mean().item():.1f}")
# build 2D mesh
from torch.distributed.device_mesh import init_device_mesh
nn=ws//lws
mesh=init_device_mesh("xpu",(nn,lws),mesh_dim_names=("replicate","shard")); log(f"mesh {nn}x{lws} built")
rep=mesh.get_group("replicate"); shd=mesh.get_group("shard"); log("got subgroups")
y=torch.ones(1024,1024,device="xpu:0"); dist.all_reduce(y,group=shd); torch.xpu.synchronize(); log(f"SHARD all_reduce OK sum={y.mean().item():.1f}")
z=torch.ones(1024,1024,device="xpu:0"); dist.all_reduce(z,group=rep); torch.xpu.synchronize(); log(f"REPLICATE all_reduce OK sum={z.mean().item():.1f} <-- the suspect")
log("ALL COLLECTIVES PASSED")
dist.barrier(); dist.destroy_process_group()
