"""Context parallel tests. Unit tests (no torchrun) + dist test for ring attention."""
import torch

from weigou.parallel.context import ring_attention, ring_attention_forward, ring_attention_backward


def test_ring_attention_forward_math():
    """ring_attention_forward produces valid attention output."""
    B, H, S, D = 2, 2, 4, 8
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    sm_scale = 1.0 / (D**0.5)

    out, lse = ring_attention_forward(q, k, v, sm_scale, is_causal=True)
    assert out.shape == (B, H, S, D)
    assert lse.shape == (B, H, S)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_ring_attention_backward_math():
    """ring_attention_backward gradients are correct shape and finite."""
    B, H, S, D = 2, 2, 4, 8
    Q = torch.randn(B, H, S, D, requires_grad=True)
    K = torch.randn(B, H, S, D, requires_grad=True)
    V = torch.randn(B, H, S, D, requires_grad=True)
    O, lse = ring_attention_forward(Q, K, V, 1.0 / (D**0.5), is_causal=True)
    dO = torch.ones_like(O)

    dQ, dK, dV = ring_attention_backward(dO, Q.detach(), K.detach(), V.detach(), O.detach(), lse, 1.0 / (D**0.5), is_causal=True)
    assert dQ.shape == Q.shape and dK.shape == K.shape and dV.shape == V.shape
    assert not torch.isnan(dQ).any()
    assert not torch.isnan(dK).any()
    assert not torch.isnan(dV).any()


def test_ring_attention_cp1():
    """With cp_size=1, ring_attention matches local attention (no ring comm)."""
    from tests.conftest import skip_if_not_distributed
    skip_if_not_distributed("run with: torchrun --nproc_per_node 1 tests/test_context_parallel.py")

    from tests.conftest import init_dist
    init_dist(tp=1, cp=1, pp=1, dp=1)

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    B, H, S, D = 1, 2, 4, 8
    q = torch.randn(B, H, S, D, device=dev, requires_grad=True)
    k = torch.randn(B, H, S, D, device=dev, requires_grad=True)
    v = torch.randn(B, H, S, D, device=dev, requires_grad=True)
    sm_scale = 1.0 / (D**0.5)

    out = ring_attention(q, k, v, sm_scale, is_causal=True)
    ref, _ = ring_attention_forward(q, k, v, sm_scale, is_causal=True)
    torch.testing.assert_close(out, ref)


if __name__ == "__main__":
    import sys
    test_ring_attention_forward_math()
    test_ring_attention_backward_math()
    if "RANK" in __import__("os").environ:
        test_ring_attention_cp1()
    print("OK")
