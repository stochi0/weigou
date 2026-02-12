"""
Tensor parallel tests. Run with:
  torchrun --nproc_per_node 2 tests/test_tensor_parallel.py
  (or 4 for tp_size=4)
"""
import os

import torch
import torch.nn as nn

import weigou.parallel.manager as pgm
from weigou.parallel.tensor import ColumnParallelLinear, RowParallelLinear
from weigou.utils import set_all_seed
from tests.conftest import init_dist

BS, SEQ, INP, OUT = 2, 4, 8, 16


def test_tp_linear_match_reference():
    """Column/Row parallel Linear must match reference forward and backward."""
    from tests.conftest import skip_if_not_distributed
    skip_if_not_distributed("run with: torchrun --nproc_per_node N tests/test_tensor_parallel.py")
    if not torch.cuda.is_available():
        print("Skip: no CUDA")
        return

    ws = int(os.environ["WORLD_SIZE"])
    init_dist(tp=ws, cp=1, pp=1, dp=1)
    set_all_seed(42)

    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    x = torch.randn(BS, SEQ, INP, device=dev, requires_grad=True)
    x_col = x.clone().detach().requires_grad_(True)
    x_row = x.chunk(ws, dim=-1)[int(os.environ["LOCAL_RANK"])].detach().requires_grad_(True)

    ref = nn.Linear(INP, OUT, bias=True, device=dev)
    col = ColumnParallelLinear(INP, OUT, bias=True, gather_output=True).to(dev)
    row = RowParallelLinear(INP, OUT, bias=True).to(dev)

    lr = int(os.environ["LOCAL_RANK"])
    col.weight = nn.Parameter(ref.weight.chunk(ws, dim=0)[lr])
    row.weight = nn.Parameter(ref.weight.chunk(ws, dim=1)[lr])
    col.bias = nn.Parameter(ref.bias.chunk(ws, dim=0)[lr])
    row.bias = nn.Parameter(ref.bias)

    y_ref = ref(x)
    y_col = col(x_col)
    y_row = row(x_row)
    torch.testing.assert_close(y_ref, y_col)
    torch.testing.assert_close(y_ref, y_row)

    g = torch.ones_like(y_ref)
    y_ref.backward(g)
    y_col.backward(g)
    y_row.backward(g)

    m = pgm.process_group_manager
    torch.testing.assert_close(ref.weight.grad.chunk(ws, dim=0)[lr], col.weight.grad)
    torch.testing.assert_close(ref.weight.grad.chunk(ws, dim=1)[lr], row.weight.grad)
    torch.testing.assert_close(x.grad, x_col.grad)
    torch.testing.assert_close(x.grad.chunk(ws, dim=-1)[lr], x_row.grad)
    torch.testing.assert_close(ref.bias.grad.chunk(ws, dim=0)[lr], col.bias.grad)
    torch.testing.assert_close(ref.bias.grad, row.bias.grad)


if __name__ == "__main__":
    test_tp_linear_match_reference()
    print("OK")
