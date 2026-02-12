"""Pipeline parallel tests. Run with: torchrun --nproc_per_node 2 tests/test_pipeline_parallel.py"""
import os

import weigou.parallel.manager as pgm
from weigou.parallel.pipeline import PipelineParallel
from tests.conftest import init_dist


def _layer_distribution(num_layers: int, pp_size: int, pp_rank: int) -> list[int]:
    """Same logic as PipelineParallel.distribute_layers."""
    per_rank = [num_layers // pp_size + (1 if i < num_layers % pp_size else 0) for i in range(pp_size)]
    start = sum(per_rank[:pp_rank])
    return list(range(start, start + per_rank[pp_rank]))


def test_layer_distribution_math():
    """Unit test: layer distribution formula."""
    assert _layer_distribution(4, 2, 0) == [0, 1]
    assert _layer_distribution(4, 2, 1) == [2, 3]
    assert _layer_distribution(5, 2, 0) == [0, 1, 2]  # rank 0 gets ceil(5/2)=3
    assert _layer_distribution(5, 2, 1) == [3, 4]
    assert _layer_distribution(3, 3, 0) == [0]
    assert _layer_distribution(3, 3, 1) == [1]
    assert _layer_distribution(3, 3, 2) == [2]


def test_pp_layer_distribution():
    """PP grid: layer distribution matches formula per rank."""
    from tests.conftest import skip_if_not_distributed
    skip_if_not_distributed("run with: torchrun --nproc_per_node 2 tests/test_pipeline_parallel.py")

    init_dist(tp=1, cp=1, pp=2, dp=1)
    pp_rank = pgm.process_group_manager.pp_rank
    pp_size = pgm.process_group_manager.pp_world_size

    num_layers = 4
    expected = _layer_distribution(num_layers, pp_size, pp_rank)
    got = PipelineParallel.distribute_layers(None, num_layers)
    assert got == expected, f"rank {pp_rank}: got {got}, expected {expected}"


if __name__ == "__main__":
    test_layer_distribution_math()
    test_pp_layer_distribution()
    print("OK")
