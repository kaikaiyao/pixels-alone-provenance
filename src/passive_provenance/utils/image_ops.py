from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_rgb_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    return image.convert("RGB")


def save_rgb_png(image: Image.Image, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean = Image.new("RGB", image.size)
    clean.putdata(list(image.getdata()))
    clean.save(path, format="PNG")


def center_crop_resize(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def normalize_real_image(source_path: str | Path, dest_path: str | Path, size: int) -> None:
    image = load_rgb_image(source_path)
    normalized = center_crop_resize(image, size)
    save_rgb_png(normalized, dest_path)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    clipped = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (clipped.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def differentiable_clip_preprocess(
    images: torch.Tensor,
    input_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    resized = F.interpolate(images, size=(input_size, input_size), mode="bicubic", align_corners=False)
    mean_tensor = torch.tensor(mean, device=resized.device, dtype=resized.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=resized.device, dtype=resized.dtype).view(1, 3, 1, 1)
    return (resized - mean_tensor) / std_tensor


def linf_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).abs().flatten(1).amax(dim=1)


def psnr_from_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = ((a - b) ** 2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(torch.full_like(mse, 1.0) / mse)


def pretty_psnr(value: float) -> float:
    if math.isinf(value):
        return 99.0
    return value
