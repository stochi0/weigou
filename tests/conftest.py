"""Shared test helpers for distributed tests. Run with: torchrun --nproc_per_node N tests/test_*.py"""
import datetime
import os
import sys

import torch
import torch.distributed as dist

from weigou.parallel import setup_process_group_manager


def skip_if_not_distributed(reason: str):
    """Skip test when not running under torchrun."""
    if "RANK" not in os.environ:
        try:
            import pytest
            pytest.skip(reason)
        except ImportError:
            raise SystemExit(reason)


def init_dist(tp: int = 1, cp: int = 1, pp: int = 1, dp: int = 1, backend: str | None = None):
    """Initialize process group and setup process group manager."""
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() and sys.platform != "darwin" else "gloo"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method="env://",
        timeout=datetime.timedelta(minutes=3),
    )
    setup_process_group_manager(tp_size=tp, cp_size=cp, pp_size=pp, dp_size=dp)
    return rank, world_size
