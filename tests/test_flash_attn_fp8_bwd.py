import math

import pytest
import torch
import torch.nn.functional as F
import flash_attn as fa


is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)


def quantize_to_fp8(x, fp8_dtype=torch.float8_e4m3fn):
    amax = x.abs().max()
    scale = (448.0 / amax).clamp(min=1e-8)
    descale = 1.0 / scale
    x_fp8 = (x * scale).to(fp8_dtype).view(torch.uint8)
    return x_fp8, scale, descale


def attention_ref(q, k, v, causal=False, dropout_p=0.0):
    """Reference attention in float32 for comparison."""
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)


def plot_fp8_histograms(x, scale, descale, fp8_dtype=torch.float8_e4m3fn, title="FP8 Quantization Debug"):
    """Debug helper: visualize quantization / dequantization effects."""
    import matplotlib.pyplot as plt

    with torch.no_grad():
        x_scaled = (x * scale).to(fp8_dtype)
        x_dequant = x_scaled.to(torch.float32) * descale

        x_np = x.flatten().cpu().float().numpy()
        x_scaled_np = x_scaled.flatten().cpu().float().numpy()
        x_dequant_np = x_dequant.flatten().cpu().float().numpy()

        vmax = float(torch.quantile(x.abs(), 0.99)) * 1.1
        bins = 80

        plt.figure(figsize=(10, 6))
        plt.hist(x_np, bins=bins, alpha=0.5, label="Original (FP32)", color="blue", range=(-vmax, vmax))
        plt.hist(x_scaled_np, bins=bins, alpha=0.5, label="Quantized (FP8 domain)", color="orange", range=(-vmax, vmax))
        plt.hist(x_dequant_np, bins=bins, alpha=0.5, label="Dequantized (FP32 after FP8)", color="green", range=(-vmax, vmax))

        plt.title(f"{title}\nscale={scale.item():.2e}, descale={descale.item():.2e}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("d", [32, 64, 128])
# @pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("seqlen", [64, 128, 256])
# @pytest.mark.parametrize("seqlen", [64])
@pytest.mark.parametrize("dropout_p", [0.0])
# @pytest.mark.parametrize("dropout_p", [0.0, 0.17])
def test_flash_attn_fp8_bwd(seqlen, d, dropout_p, causal, dtype):
    if not is_sm8x:
        pytest.skip("FP8 bwd only supported on SM80+")
    device = "cuda"
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 4

    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True)

    q_fp8, q_scale, q_descale = quantize_to_fp8(q)
    k_fp8, k_scale, k_descale = quantize_to_fp8(k)
    v_fp8, v_scale, v_descale = quantize_to_fp8(v)
    o_descale = torch.tensor([1.0], device=device)

    softmax_scale = d ** -0.5

    # --- reference fwd + bwd in float32
    out_ref = attention_ref(q.float(), k.float(), v.float(), causal=causal, dropout_p=dropout_p)
    g = torch.randn_like(out_ref)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)

    # --- fp8 fwd
    out_fp8 = fa.fwd(
        q_fp8, k_fp8, v_fp8,
        p_dropout=dropout_p,
        softmax_scale=softmax_scale,
        is_causal=causal,
    )

    print(f"Output max diff: {(out_fp8.float() - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out_fp8.float() - out_ref).abs().mean().item()}")

    # --- fp8 bwd
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    fa.flash_attn_backward_fp8(                                                                                                                                                                                             
        g, q_fp8, k_fp8, v_fp8, out_fp8,                                                                                                                                                                                    
        softmax_lse=torch.empty(batch_size, nheads, seqlen, device=device, dtype=dtype),                                                                                                                                    
        dq=dq, dk=dk, dv=dv,                                                                                                                                                                                                
        dropout_p=dropout_p,                                                                                                                                                                                                
        softmax_scale=softmax_scale,
        causal=causal,
        q_descale=torch.tensor([q_descale], device=device),
        k_descale=torch.tensor([k_descale], device=device),
        v_descale=torch.tensor([v_descale], device=device),
        o_descale=torch.tensor([1.0], device=device),
        q_scale=torch.tensor([q_scale], device=device),
        k_scale=torch.tensor([k_scale], device=device),
        v_scale=torch.tensor([v_scale], device=device),
    )

    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
    print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")

    torch.testing.assert_close(dq, dq_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(dk, dk_ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(dv, dv_ref, rtol=5e-2, atol=5e-2)
