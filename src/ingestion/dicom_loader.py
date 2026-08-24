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
    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    min_v = float(image.min())
    max_v = float(image.max())
    if max_v <= min_v:
        return np.zeros_like(image, dtype=np.float32)

    if ScaleIntensityRange and EnsureType:
        scaler = ScaleIntensityRange(a_min=min_v, a_max=max_v, b_min=0.0, b_max=1.0, clip=True)
        image = scaler(image)
        image = EnsureType(dtype=np.float32)(image)
        return np.asarray(image, dtype=np.float32)

    return (image - min_v) / (max_v - min_v)


def load_dicom(dicom_path: str | Path, phi_tags: list[str] | None = None) -> DICOMLoadResult:
    dataset = pydicom.dcmread(str(dicom_path))
    stripped = strip_phi_tags(dataset, phi_tags or DEFAULT_PHI_TAGS)
    warnings: list[str] = []

    if (
        "PixelData" not in dataset
        and "FloatPixelData" not in dataset
        and "DoubleFloatPixelData" not in dataset
    ):
        raise ValueError("DICOM contains no pixel data")

    number_of_frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    if number_of_frames != 1:
        raise ValueError("Multi-frame DICOM is not supported; upload one chest radiograph at a time")

    pixels = dataset.pixel_array
    samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    if pixels.ndim == 2:
        image = pixels
    elif pixels.ndim == 3 and samples_per_pixel in {3, 4} and pixels.shape[-1] >= 3:
        image = np.asarray(pixels[..., :3], dtype=np.float32).mean(axis=-1)
        warnings.append("Color DICOM was converted to grayscale for chest-radiograph processing.")
    else:
        raise ValueError(f"Unsupported pixel shape: {pixels.shape}")

    image_np = _normalize_image(np.asarray(image))
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        image_np = 1.0 - image_np
        warnings.append("MONOCHROME1 intensity was inverted for display.")

    rows, columns = image_np.shape
    if min(rows, columns) < 256:
        warnings.append("Image resolution is below 256 pixels on one axis; model output may be unreliable.")
    if float(image_np.std()) < 0.01:
        warnings.append("Image has very low contrast; analysis may be unreliable.")

    modality = str(getattr(dataset, "Modality", "")).upper()
    sop_class_uid = str(getattr(dataset, "SOPClassUID", ""))
    if modality not in {"CR", "DX"} or sop_class_uid == "1.2.840.10008.5.1.4.1.1.7":
        warnings.append(
            "This is not an original CR/DX acquisition (for example, a secondary capture); "
            "use it for software demonstration only."
        )

    return DICOMLoadResult(image_np=image_np, stripped_tags=stripped, warnings=warnings)


def to_rgb(image_np: np.ndarray) -> np.ndarray:
    if image_np.ndim != 2:
        raise ValueError("Expected grayscale array")
    rgb = np.stack([image_np, image_np, image_np], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
