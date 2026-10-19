from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.transforms import functional as tvf

from passive_provenance.config import resolve_artifact_relative
from passive_provenance.matrix import get_verifier_spec
from passive_provenance.models.d1 import load_open_clip_encoder
from passive_provenance.utils.image_ops import load_rgb_image, pil_to_tensor
from passive_provenance.utils.io import ensure_parent


DEFAULT_CLIP_REAL_PROMPTS = (
    "a real photograph",
    "a genuine camera photo",
    "a natural unedited image",
)

DEFAULT_CLIP_FAKE_PROMPTS = (
    "an AI-generated image",
    "a synthetic diffusion image",
    "a fake computer-generated photo",
)


@dataclass(frozen=True)
class ExternalDetectorMetadata:
    backend_name: str
    class_names: tuple[str, ...]
    extras: dict[str, object]


class ExternalDetectorAdapter:
    def __init__(self, verifier_id: str, class_names: Sequence[str]) -> None:
        self.verifier_id = verifier_id
        self.class_names = tuple(class_names)

    def predict_probabilities(self, image_paths: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError

    def predict_tensor_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def metadata(self) -> ExternalDetectorMetadata:
        return ExternalDetectorMetadata(
            backend_name=self.verifier_id,
            class_names=self.class_names,
            extras={},
        )


class NotImplementedExternalDetectorAdapter(ExternalDetectorAdapter):
    def __init__(self, verifier_id: str, class_names: Sequence[str], note: str) -> None:
        super().__init__(verifier_id, class_names)
        self.note = note

    def predict_probabilities(self, image_paths: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError(self.note)

    def metadata(self) -> ExternalDetectorMetadata:
        return ExternalDetectorMetadata(
            backend_name=self.verifier_id,
            class_names=self.class_names,
            extras={"note": self.note},
        )


class ClipPromptDetectorAdapter(ExternalDetectorAdapter):
    def __init__(
        self,
        verifier_id: str,
        class_names: Sequence[str],
        *,
        model_name: str,
        pretrained: str,
        input_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        real_prompts: Sequence[str],
        fake_prompts: Sequence[str],
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__(verifier_id, class_names)
        if tuple(self.class_names) != ("real", "fake"):
            raise ValueError(f"{verifier_id} currently expects binary class_names=('real', 'fake').")
        self.model_name = model_name
        self.pretrained = pretrained
        self.input_size = int(input_size)
        self.mean = tuple(float(item) for item in mean)
        self.std = tuple(float(item) for item in std)
        self.real_prompts = tuple(real_prompts)
        self.fake_prompts = tuple(fake_prompts)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)

        import open_clip

        self.model = load_open_clip_encoder(model_name, pretrained).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text_features = self._build_text_features()
        print(
            f"[{self.verifier_id}] initialized clip prompt detector on device={self.device.type} "
            f"with prompts real={len(self.real_prompts)} fake={len(self.fake_prompts)}",
            flush=True,
        )

    def _preprocess_tensors(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.to(self.device)
        mean = torch.tensor(self.mean, device=self.device, dtype=batch.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=self.device, dtype=batch.dtype).view(1, 3, 1, 1)
        resized = F.interpolate(batch, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        return (resized - mean) / std

    def _load_path_batch(self, image_paths: Sequence[str]) -> torch.Tensor:
        return torch.stack([pil_to_tensor(load_rgb_image(path)) for path in image_paths], dim=0)

    def _encode_prompt_bank(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        pooled = text_features.mean(dim=0, keepdim=True)
        return F.normalize(pooled, dim=-1)

    def _build_text_features(self) -> torch.Tensor:
        banks = [
            self._encode_prompt_bank(self.real_prompts),
            self._encode_prompt_bank(self.fake_prompts),
        ]
        return torch.cat(banks, dim=0)

    def _predict_batches(self, images: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        total_batches = max(1, (images.shape[0] + self.batch_size - 1) // self.batch_size)
        with torch.inference_mode():
            for batch_index, start in enumerate(range(0, images.shape[0], self.batch_size), start=1):
                image_batch = self._preprocess_tensors(images[start : start + self.batch_size])
                image_features = self.model.encode_image(image_batch)
                image_features = F.normalize(image_features, dim=-1)
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * image_features @ self.text_features.T
                outputs.append(torch.softmax(logits, dim=1).detach().cpu())
                if total_batches > 1 and (batch_index == 1 or batch_index == total_batches or batch_index % 5 == 0):
                    print(
                        f"[{self.verifier_id}] processed batch {batch_index}/{total_batches}",
                        flush=True,
                    )
        return torch.cat(outputs, dim=0)

    def predict_probabilities(self, image_paths: Sequence[str]) -> torch.Tensor:
        print(
            f"[{self.verifier_id}] starting clean inference for {len(image_paths)} images "
            f"with batch_size={self.batch_size} on device={self.device.type}",
            flush=True,
        )
        outputs: list[torch.Tensor] = []
        total_batches = max(1, (len(image_paths) + self.batch_size - 1) // self.batch_size)
        for batch_index, start in enumerate(range(0, len(image_paths), self.batch_size), start=1):
            batch_paths = image_paths[start : start + self.batch_size]
            outputs.append(self._predict_batches(self._load_path_batch(batch_paths)))
            if total_batches > 1 and (batch_index == 1 or batch_index == total_batches or batch_index % 5 == 0):
                print(
                    f"[{self.verifier_id}] loaded path batch {batch_index}/{total_batches}",
                    flush=True,
                )
        return torch.cat(outputs, dim=0)

    def predict_tensor_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return self._predict_batches(images)

    def metadata(self) -> ExternalDetectorMetadata:
        return ExternalDetectorMetadata(
            backend_name="clip_prompt_detector",
            class_names=self.class_names,
            extras={
                "model_name": self.model_name,
                "pretrained": self.pretrained,
                "input_size": self.input_size,
                "real_prompt_count": len(self.real_prompts),
                "fake_prompt_count": len(self.fake_prompts),
            },
        )


def external_model_path(config: dict[str, object], verifier_id: str, filename: str) -> Path:
    return resolve_artifact_relative(config, f"artifacts/external_models/{verifier_id}/{filename}")


def is_html_like_file(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as handle:
        prefix = handle.read(256).lstrip()
    return prefix.startswith((b"<", b"<!DOCTYPE html", b"<html"))


def canonical_download_url(url: str) -> str:
    if "dropbox.com" in url and "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    return url


def download_to_path(url: str, destination: Path) -> Path:
    ensure_parent(destination)
    if destination.exists() and not is_html_like_file(destination):
        return destination
    if not url.strip():
        raise RuntimeError(
            f"Missing external-detector weights at {destination}. Populate the file locally or set a direct weight_url."
        )
    if destination.exists():
        destination.unlink()
    resolved_url = canonical_download_url(url)
    with urlopen(resolved_url) as response, destination.open("wb") as handle:
        handle.write(response.read())
    if is_html_like_file(destination):
        raise RuntimeError(f"Downloaded HTML instead of model weights from {resolved_url}.")
    return destination


class CnnDetectionAdapter(ExternalDetectorAdapter):
    def __init__(
        self,
        verifier_id: str,
        class_names: Sequence[str],
        *,
        weight_path: Path,
        weight_url: str,
        batch_size: int,
        device: torch.device,
        resize_size: int = 256,
        crop_size: int = 224,
    ) -> None:
        super().__init__(verifier_id, class_names)
        if tuple(self.class_names) != ("real", "fake"):
            raise ValueError(f"{verifier_id} currently expects binary class_names=('real', 'fake').")
        self.weight_path = download_to_path(weight_url, weight_path)
        self.weight_url = weight_url
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)

        self.model = resnet50(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        state_dict = torch.load(self.weight_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        print(
            f"[{self.verifier_id}] initialized cnnspot on device={self.device.type} "
            f"with weights={self.weight_path.name}",
            flush=True,
        )

    def _preprocess_tensors(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.to(self.device)
        height, width = int(batch.shape[-2]), int(batch.shape[-1])
        short_edge = max(1, min(height, width))
        scale = float(self.resize_size) / float(short_edge)
        resized_height = max(self.crop_size, int(round(height * scale)))
        resized_width = max(self.crop_size, int(round(width * scale)))
        resized = F.interpolate(batch, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
        top = max((resized_height - self.crop_size) // 2, 0)
        left = max((resized_width - self.crop_size) // 2, 0)
        cropped = resized[:, :, top : top + self.crop_size, left : left + self.crop_size]
        if cropped.shape[-2:] != (self.crop_size, self.crop_size):
            cropped = F.interpolate(cropped, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=cropped.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=cropped.dtype).view(1, 3, 1, 1)
        return (cropped - mean) / std

    def _predict_batches(self, images: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        total_batches = max(1, (images.shape[0] + self.batch_size - 1) // self.batch_size)
        with torch.inference_mode():
            for batch_index, start in enumerate(range(0, images.shape[0], self.batch_size), start=1):
                image_batch = self._preprocess_tensors(images[start : start + self.batch_size])
                fake_prob = torch.sigmoid(self.model(image_batch).flatten())
                probabilities = torch.stack([1.0 - fake_prob, fake_prob], dim=1)
                outputs.append(probabilities.detach().cpu())
                if total_batches > 1 and (batch_index == 1 or batch_index == total_batches or batch_index % 5 == 0):
                    print(
                        f"[{self.verifier_id}] processed batch {batch_index}/{total_batches}",
                        flush=True,
                    )
        return torch.cat(outputs, dim=0)

    def predict_probabilities(self, image_paths: Sequence[str]) -> torch.Tensor:
        print(
            f"[{self.verifier_id}] starting clean inference for {len(image_paths)} images "
            f"with batch_size={self.batch_size} on device={self.device.type}",
            flush=True,
        )
        outputs: list[torch.Tensor] = []
        total_batches = max(1, (len(image_paths) + self.batch_size - 1) // self.batch_size)
        for batch_index, start in enumerate(range(0, len(image_paths), self.batch_size), start=1):
            tensors = []
            for path in image_paths[start : start + self.batch_size]:
                image = load_rgb_image(path)
                image = tvf.resize(image, self.resize_size)
                image = tvf.center_crop(image, [self.crop_size, self.crop_size])
                tensors.append(tvf.to_tensor(image))
            outputs.append(self._predict_batches(torch.stack(tensors, dim=0)))
            if total_batches > 1 and (batch_index == 1 or batch_index == total_batches or batch_index % 5 == 0):
                print(
                    f"[{self.verifier_id}] loaded path batch {batch_index}/{total_batches}",
                    flush=True,
                )
        return torch.cat(outputs, dim=0)

    def predict_tensor_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return self._predict_batches(images)

    def metadata(self) -> ExternalDetectorMetadata:
        return ExternalDetectorMetadata(
            backend_name="cnn_detection",
            class_names=self.class_names,
            extras={
                "weight_path": str(self.weight_path),
                "resize_size": self.resize_size,
                "crop_size": self.crop_size,
            },
        )


class DireDetectionAdapter(ExternalDetectorAdapter):
    def __init__(
        self,
        verifier_id: str,
        class_names: Sequence[str],
        *,
        weight_path: Path,
        weight_url: str,
        arch: str,
        batch_size: int,
        device: torch.device,
        resize_size: int = 256,
        crop_size: int = 224,
        aug_norm: bool = True,
    ) -> None:
        super().__init__(verifier_id, class_names)
        if tuple(self.class_names) != ("real", "fake"):
            raise ValueError(f"{verifier_id} currently expects binary class_names=('real', 'fake').")
        self.weight_path = download_to_path(weight_url, weight_path)
        self.weight_url = weight_url
        self.arch = str(arch)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)
        self.aug_norm = bool(aug_norm)

        model_builder = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
            "resnet152": resnet152,
        }.get(self.arch)
        if model_builder is None:
            raise ValueError(f"Unsupported DIRE arch={self.arch!r}.")
        self.model = model_builder(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        state_dict = torch.load(self.weight_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        print(
            f"[{self.verifier_id}] initialized dire on device={self.device.type} "
            f"with arch={self.arch} weights={self.weight_path.name} aug_norm={self.aug_norm}",
            flush=True,
        )

    def _preprocess_tensors(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.to(self.device)
        height, width = int(batch.shape[-2]), int(batch.shape[-1])
        short_edge = max(1, min(height, width))
        scale = float(self.resize_size) / float(short_edge)
        resized_height = max(self.crop_size, int(round(height * scale)))
        resized_width = max(self.crop_size, int(round(width * scale)))
        resized = F.interpolate(batch, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
        top = max((resized_height - self.crop_size) // 2, 0)
        left = max((resized_width - self.crop_size) // 2, 0)
        cropped = resized[:, :, top : top + self.crop_size, left : left + self.crop_size]
        if cropped.shape[-2:] != (self.crop_size, self.crop_size):
            cropped = F.interpolate(cropped, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
        if not self.aug_norm:
            return cropped
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=cropped.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=cropped.dtype).view(1, 3, 1, 1)
        return (cropped - mean) / std

    def _predict_batches(self, images: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        total_batches = max(1, (images.shape[0] + self.batch_size - 1) // self.batch_size)
        with torch.inference_mode():
            for batch_index, start in enumerate(range(0, images.shape[0], self.batch_size), start=1):
                image_batch = self._preprocess_tensors(images[start : start + self.batch_size])
                fake_prob = torch.sigmoid(self.model(image_batch).flatten())
                probabilities = torch.stack([1.0 - fake_prob, fake_prob], dim=1)
                outputs.append(probabilities.detach().cpu())
                if total_batches > 1 and (batch_index == 1 or batch_index == total_batches or batch_index % 5 == 0):
                    print(
                        f"[{self.verifier_id}] processed batch {batch_index}/{total_batches}",
                        flush=True,
                    )
        return torch.cat(outputs, dim=0)

    def predict_probabilities(self, image_paths: Sequence[str]) -> torch.Tensor:
        print(
            f"[{self.verifier_id}] starting clean inference for {len(image_paths)} images "
            f"with batch_size={self.batch_size} on device={self.device.type}",
            flush=True,
        )
        outputs: list[torch.Tensor] = []
        total_batches = max(1, (len(image_paths) + self.batch_size - 1) // self.batch_size)
        for batch_index, start in enumerate(range(0, len(image_paths), self.batch_size), start=1):
            tensors = []
            for path in image_paths[start : start + self.batch_size]:
                image = load_rgb_image(path)
                image = tvf.resize(image, self.resize_size)
                image = tvf.center_crop(image, [self.crop_size, self.crop_size])
                tensors.append(tvf.to_tensor(image))
            outputs.append(self._predict_batches(torch.stack(tensors, dim=0)))
            if total_batches > 1 and (batch_index == 1 or batch_index == total_batches or batch_index % 5 == 0):
                print(
                    f"[{self.verifier_id}] loaded path batch {batch_index}/{total_batches}",
                    flush=True,
                )
        return torch.cat(outputs, dim=0)

    def predict_tensor_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return self._predict_batches(images)

    def metadata(self) -> ExternalDetectorMetadata:
        return ExternalDetectorMetadata(
            backend_name="dire_detection",
            class_names=self.class_names,
            extras={
                "weight_path": str(self.weight_path),
                "arch": self.arch,
                "resize_size": self.resize_size,
                "crop_size": self.crop_size,
                "aug_norm": self.aug_norm,
            },
        )


def build_external_detector(
    config: dict[str, object],
    verifier_id: str,
    *,
    device: torch.device | str | None = None,
) -> ExternalDetectorAdapter:
    verifier = get_verifier_spec(config, verifier_id)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    detector_config = dict(config.get("external_detectors", {}))
    batch_size = int(detector_config.get("batch_size", config["feature_cache"]["batch_size"]))
    if verifier.head_type == "clip_detector":
        clip_config = dict(detector_config.get("clip_detector", {}))
        model_name = str(detector_config.get("clip_model_name", config["d1"]["clip_model_name"]))
        pretrained = str(detector_config.get("clip_pretrained", config["d1"]["clip_pretrained"]))
        real_prompts = tuple(clip_config.get("real_prompts", DEFAULT_CLIP_REAL_PROMPTS))
        fake_prompts = tuple(clip_config.get("fake_prompts", DEFAULT_CLIP_FAKE_PROMPTS))
        return ClipPromptDetectorAdapter(
            verifier_id=verifier_id,
            class_names=verifier.class_names,
            model_name=model_name,
            pretrained=pretrained,
            input_size=int(config["data"]["clip_input_size"]),
            mean=tuple(config["data"]["clip_mean"]),
            std=tuple(config["data"]["clip_std"]),
            real_prompts=real_prompts,
            fake_prompts=fake_prompts,
            batch_size=batch_size,
            device=device,
        )
    if verifier.head_type == "cnnspot":
        cnnspot_config = dict(detector_config.get("cnnspot", {}))
        weight_filename = str(cnnspot_config.get("weight_filename", "blur_jpg_prob0.5.pth"))
        return CnnDetectionAdapter(
            verifier_id,
            verifier.class_names,
            weight_path=external_model_path(config, verifier_id, weight_filename),
            weight_url=str(
                cnnspot_config.get(
                    "weight_url",
                    "https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=0",
                )
            ),
            batch_size=int(cnnspot_config.get("batch_size", batch_size)),
            device=device,
            resize_size=int(cnnspot_config.get("resize_size", 256)),
            crop_size=int(cnnspot_config.get("crop_size", 224)),
        )
    if verifier.head_type == "dire":
        dire_config = dict(detector_config.get("dire", {}))
        weight_filename = str(dire_config.get("weight_filename", "lsun_adm.pth"))
        return DireDetectionAdapter(
            verifier_id,
            verifier.class_names,
            weight_path=external_model_path(config, verifier_id, weight_filename),
            weight_url=str(dire_config.get("weight_url", "")).strip(),
            arch=str(dire_config.get("arch", "resnet50")),
            batch_size=int(dire_config.get("batch_size", batch_size)),
            device=device,
            resize_size=int(dire_config.get("resize_size", 256)),
            crop_size=int(dire_config.get("crop_size", 224)),
            aug_norm=bool(dire_config.get("aug_norm", True)),
        )
    raise ValueError(f"Unsupported external detector head_type={verifier.head_type} for verifier_id={verifier_id}.")
