from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_window(
    *, channels: int, window_size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_per_image(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Standard windowed SSIM for same-shaped RGB tensors in [0, 1]."""
    if original.shape != adversarial.shape or original.ndim != 4:
        raise ValueError("SSIM inputs must be same-shaped NCHW tensors.")
    channels = original.shape[1]
    window = gaussian_window(
        channels=channels,
        window_size=window_size,
        sigma=sigma,
        device=original.device,
        dtype=original.dtype,
    )
    padding = window_size // 2
    mu_x = F.conv2d(original, window, padding=padding, groups=channels)
    mu_y = F.conv2d(adversarial, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(original.square(), window, padding=padding, groups=channels) - mu_x_sq
    sigma_y = F.conv2d(adversarial.square(), window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(original * adversarial, window, padding=padding, groups=channels) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).flatten(1).mean(dim=1)
