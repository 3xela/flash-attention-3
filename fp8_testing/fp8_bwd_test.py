import torch
import pytest

import flash_attn_2_cuda as fa


def quantize_to_fp8(x: torch.Tensor, fp8_dtype=torch.float8_e4m3fn):
    """
    Quantize tensor to FP8 with scale/descale factors.
    Returns quantized tensor (uint8 view), scale, descale.
    """
    # Get maximum absolute value
    amax = x.abs().max()
    # Safe guard against zeros
    scale = (448.0 / amax).clamp(min=1e-8)  # 448 is max for E4M3
    descale = 1.0 / scale

    # Scale to fit FP8 dynamic range
    x_scaled = (x * scale).to(fp8_dtype)
    # Store quantized as torch.uint8 view (matches your kernel interface)
    x_fp8 = x_scaled.view(torch.uint8)

    return x_fp8, scale, descale


def run_flash_attn_fp8(batch=2, seqlen_q=64, seqlen_k=64, nheads=4, head_dim=32, dropout=0.0, causal=False):
    device = "cuda"
    dtype = torch.float32

    q = torch.randn(batch, seqlen_q, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seqlen_k, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seqlen_k, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)

    # Quantize Q, K, V to FP8
    q_fp8, q_scale, q_descale = quantize_to_fp8(q)
    k_fp8, k_scale, k_descale = quantize_to_fp8(k)
    v_fp8, v_scale, v_descale = quantize_to_fp8(v)

    # O descale is usually taken from fwd pass, here set to 1
    o_descale = torch.tensor([1.0], device=device)

    # Forward pass
    scale = head_dim ** -0.5
    out = fa.fwd(q_fp8, k_fp8, v_fp8, p_dropout=dropout, softmax_scale=scale, is_causal=causal)

    # Reference
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=causal)

    # Accuracy check (looser because FP8)
    torch.testing.assert_close(out, ref_out, rtol=5e-2, atol=5e-2)

    # Backward
    dout = torch.randn_like(out)
    out.backward(dout, retain_graph=True)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(ref_out, (q, k, v), dout, retain_graph=True)

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    fa.bwd(
        dout,
        q_fp8,
        k_fp8,
        v_fp8,
        out,
        torch.empty(batch, nheads, seqlen_q, device=device, dtype=dtype),
        dq,
        dk,
        dv,
        None,
        dropout,
        scale,
        causal,
        -1,
        -1,
        0.0,
        False,
        None,
        None,
        torch.tensor([q_descale], device=device),
        torch.tensor([k_descale], device=device),
        torch.tensor([v_descale], device=device),
        o_descale,
        torch.tensor([q_scale], device=device),
        torch.tensor([k_scale], device=device),
        torch.tensor([v_scale], device=device),
    )

    # Gradient accuracy
    torch.testing.assert_close(dq, dq_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(dk, dk_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(dv, dv_ref, rtol=5e-2, atol=5e-2)


def test_flash_attn_fp8_quant():
    run_flash_attn_fp8()


if __name__ == "__main__":
    pytest.main([__file__])
 