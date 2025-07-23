from __future__ import annotations

import os, torch.distributed as dist, torch

__all__ = ["init_ddp", "is_main_process"]

def init_ddp() -> int:
    """Initialise torch.distributed if not yet initialised.

    Returns
    -------
    int
        local_rank id (GPU index) if on CUDA, else 0.
    """
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0 