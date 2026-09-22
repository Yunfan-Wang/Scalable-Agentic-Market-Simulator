"""torchrun setup, exact validation sharding and collective finite checks."""

import os
from datetime import timedelta
from dataclasses import dataclass
import torch
from torch import distributed as dist
from torch.utils.data import Sampler


@dataclass
class Context:
    rank: int
    world_size: int
    device: torch.device

    @classmethod
    def initialize(cls, device="auto"):
        size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.set_device(local_rank)
            selected = torch.device("cuda", local_rank)
        else:
            selected = torch.device("cpu")
        if size > 1 and not dist.is_initialized():
            dist.init_process_group(
                "nccl" if selected.type == "cuda" else "gloo",
                init_method=os.environ.get("SAMS_INIT_METHOD", "env://"),
                rank=rank,
                world_size=size,
                timeout=timedelta(seconds=120),
            )
        return cls(rank, size, selected)

    def all_finite(self, value):
        flag = torch.tensor(int(bool(value)), device=self.device)
        if self.world_size > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class EvaluationSampler(Sampler):
    """Strided, non-padding shards: each validation example is counted exactly once."""

    def __init__(self, dataset, rank=0, world_size=1):
        self.indices = range(rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
