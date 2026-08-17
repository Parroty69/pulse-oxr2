from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

try:
    from monai.transforms import EnsureType, ScaleIntensityRange
except Exception:  # pragma: no cover - optional fallback
    EnsureType = None
    ScaleIntensityRange = None


@dataclass
class DICOMLoadResult:
    image_np: np.ndarray
    stripped_tags: list[str]
    warnings: list[str]


DEFAULT_PHI_TAGS = [
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "InstitutionName",
]


def strip_phi_tags(dataset: pydicom.Dataset, phi_tags: list[str]) -> list[str]:
    stripped: list[str] = []
    for tag_name in phi_tags:
        if hasattr(dataset, tag_name):
            setattr(dataset, tag_name, "")
            stripped.append(tag_name)
    return stripped


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if ScaleIntensityRange and EnsureType:
        scaler = ScaleIntensityRange(a_min=float(image.min()), a_max=float(image.max()), b_min=0.0, b_max=1.0, clip=True)
        image = scaler(image)
        image = EnsureType(dtype=np.float32)(image)
        return np.asarray(image, dtype=np.float32)

    min_v = float(image.min())
    max_v = float(image.max())
    if max_v > min_v:
        image = (image - min_v) / (max_v - min_v)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image


def load_dicom(dicom_path: str | Path, phi_tags: list[str] | None = None) -> DICOMLoadResult:
    dataset = pydicom.dcmread(str(dicom_path))
    stripped = strip_phi_tags(dataset, phi_tags or DEFAULT_PHI_TAGS)

    if not hasattr(dataset, "pixel_array"):
        raise ValueError("DICOM has no pixel_array")

    pixels = dataset.pixel_array
    if pixels.ndim == 2:
        image = pixels
    elif pixels.ndim == 3:
        image = pixels[..., 0]
    else:
        raise ValueError(f"Unsupported pixel shape: {pixels.shape}")

    image_np = _normalize_image(np.asarray(image))
    return DICOMLoadResult(image_np=image_np, stripped_tags=stripped, warnings=[])


def to_rgb(image_np: np.ndarray) -> np.ndarray:
    if image_np.ndim != 2:
        raise ValueError("Expected grayscale array")
    rgb = np.stack([image_np, image_np, image_np], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
