import torch
import matplotlib.pyplot as plt

def plot_fp8_histograms(x: torch.Tensor, scale: torch.Tensor, descale: torch.Tensor, fp8_dtype=torch.float8_e4m3fn, title="FP8 Quantization Debug"):
    """
    Visualizes quantization and dequantization effects for FP8 tensors.
    - x: original tensor (float16 or float32)
    - scale, descale: scale factors used in quantize_to_fp8
    - fp8_dtype: torch.float8_e4m3fn or torch.float8_e5m2
    """

    with torch.no_grad():
        # Quantize/dequantize roundtrip
        x_scaled = (x * scale).to(fp8_dtype)
        x_dequant = (x_scaled.to(torch.float32) * descale)

        # CPU copies for plotting
        x_np = x.flatten().cpu().float().numpy()
        x_scaled_np = x_scaled.flatten().cpu().float().numpy()
        x_dequant_np = x_dequant.flatten().cpu().float().numpy()

        # Histogram range auto-fit (clip extreme outliers)
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
