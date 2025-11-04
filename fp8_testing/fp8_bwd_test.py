import torch
import pytest
import flash_attn as fa


def quantize_to_fp8(x: torch.Tensor, fp8_dtype=torch.float8_e4m3fn):
    amax = x.abs().max()
    scale = (448.0 / amax).clamp(min=1e-8)
    descale = 1.0 / scale
    x_fp8 = (x * scale).to(fp8_dtype).view(torch.uint8)
    return x_fp8, scale, descale


def run_flash_attn_compare_fp8_vs_fp16(
    batch=2, seqlen_q=64, seqlen_k=64, nheads=4, head_dim=32, dropout=0.0, causal=False
):
    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(42)

    # --- allocate activations
    q = torch.randn(batch, seqlen_q, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seqlen_k, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seqlen_k, nheads, head_dim, device=device, dtype=dtype, requires_grad=True)

    # --- quantize for fp8 version
    q_fp8, q_scale, q_descale = quantize_to_fp8(q)
    k_fp8, k_scale, k_descale = quantize_to_fp8(k)
    v_fp8, v_scale, v_descale = quantize_to_fp8(v)
    o_descale = torch.tensor([1.0], device=device)

    scale = head_dim ** -0.5
    dropout_p = dropout

    # --- forward reference (fp16/fp32)
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=dropout_p, is_causal=causal
    )

    # --- forward fp8 (if fwd kernel exists)
    out_fp8 = fa.fwd(
        q_fp8, k_fp8, v_fp8,
        p_dropout=dropout_p,
        softmax_scale=scale,
        is_causal=causal
    )

    # --- check forward accuracy
    torch.testing.assert_close(out_fp8, out_ref, rtol=5e-2, atol=5e-2)

    # --- backward reference (fp16/fp32)
    dout = torch.randn_like(out_ref)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref, (q, k, v), dout, retain_graph=True
    )

    # --- backward fp8
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    fa.bwd_fp8(
        dout,
        q_fp8, k_fp8, v_fp8,
        out_fp8,
        torch.empty(batch, nheads, seqlen_q, device=device, dtype=dtype),
        dq, dk, dv,
        None,
        dropout_p, scale, causal,
        -1, -1, 0.0, False, None, None,
        torch.tensor([q_descale], device=device),
        torch.tensor([k_descale], device=device),
        torch.tensor([v_descale], device=device),
        o_descale,
        torch.tensor([q_scale], device=device),
        torch.tensor([k_scale], device=device),
        torch.tensor([v_scale], device=device),
    )

    # --- compare fp8 gradients vs fp16 gradients
    print("\nmax |dq_fp8 - dq_ref|:", (dq - dq_ref).abs().max().item())
    print("max |dk_fp8 - dk_ref|:", (dk - dk_ref).abs().max().item())
    print("max |dv_fp8 - dv_ref|:", (dv - dv_ref).abs().max().item())

    torch.testing.assert_close(dq, dq_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(dk, dk_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(dv, dv_ref, rtol=5e-2, atol=5e-2)


def test_flash_attn_bwd_fp8_vs_fp16():
    run_flash_attn_compare_fp8_vs_fp16()


if __name__ == "__main__":
    pytest.main([__file__])
