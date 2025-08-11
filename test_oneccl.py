from mpi4py import MPI
import os
import socket
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl

def main():
    # DDP: Set environmental variables used by PyTorch
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()
    LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID', os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)
    print(f"DDP: Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. {MASTER_ADDR}")

    # DDP: initialize distributed communication with ccl backend
    torch.distributed.init_process_group(backend='ccl', init_method='env://', rank=int(RANK), world_size=int(SIZE))
    print("DDP: Process group initialized.")

    # DDP: pin GPU to local rank.
    torch.xpu.set_device(int(LOCAL_RANK))
    device = torch.device('xpu')
    print(f"DDP: Rank {RANK} is on device {device}.")

    # Create a tensor and perform an all-reduce operation
    tensor = torch.ones(1, device=device) * (RANK + 1)
    print(f"DDP: Rank {RANK} has tensor with value {tensor.item()}.")

    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    print(f"DDP: Rank {RANK} has tensor with value {tensor.item()} after all-reduce.")

    # Check if the all-reduce was successful
    expected_sum = sum(range(1, SIZE + 1))
    if tensor.item() == expected_sum:
        print(f"SUCCESS: Rank {RANK} has the correct sum.")
    else:
        print(f"FAILURE: Rank {RANK} has incorrect sum. Expected {expected_sum}, got {tensor.item()}.")
        exit(1)

    # DDP: cleanup
    torch.distributed.destroy_process_group()
    print("DDP: Process group destroyed.")

if __name__ == "__main__":
    main()
