"""
Data parallel tests. Run with:
  torchrun --nproc_per_node 2 tests/test_data_parallel.py
"""
import os

import torch
import torch.nn as nn

import weigou.parallel.manager as pgm
from weigou.parallel.data import DataParallelNaive
from weigou.utils import set_all_seed
from tests.conftest import init_dist


def test_dp_grad_sync():
    """DataParallelNaive should average gradients across ranks."""
    from tests.conftest import skip_if_not_distributed
    skip_if_not_distributed("run with: torchrun --nproc_per_node 2 tests/test_data_parallel.py")
    if not torch.cuda.is_available():
        print("Skip: CUDA required")
        return

    init_dist(tp=1, cp=1, pp=1, dp=2)
    set_all_seed(42)

    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    x = torch.randn(2, 4, device=dev)
    ref = nn.Linear(4, 8, bias=True).to(dev)
    dp = DataParallelNaive(nn.Linear(4, 8, bias=True).to(dev))

    # Sync weights so both ranks identical
    for p_dp, p_ref in zip(dp.parameters(), ref.parameters()):
        p_dp.data.copy_(p_ref.data)

    y_ref = ref(x)
    y_dp = dp(x)
    y_ref.backward(torch.ones_like(y_ref))
    y_dp.backward(torch.ones_like(y_dp))

    for p_dp, p_ref in zip(dp.parameters(), ref.parameters()):
        torch.testing.assert_close(p_dp.grad, p_ref.grad, msg="DP grad should match single-rank after all-reduce")


if __name__ == "__main__":
    test_dp_grad_sync()
    print("OK")
