from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def prepare_display_image(image: Any) -> np.ndarray:
    """Convert a decoded DICOM image to an RGB uint8 array for the web UI."""

    if image is None:
        raise ValueError("Pipeline result is missing the decoded DICOM image")

    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("Decoded DICOM image is empty")

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    elif array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Unsupported display image shape: {array.shape}")

    if np.issubdtype(array.dtype, np.bool_):
        array = array.astype(np.uint8) * 255
    elif np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        if float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
            array = array * 255.0

    return np.clip(array, 0, 255).astype(np.uint8)


def render_overlay(result: dict, enabled_indices: set[int]) -> Image.Image:
    image = Image.fromarray(prepare_display_image(result.get("display_image")))
    draw = ImageDraw.Draw(image)
    for idx, mask_obj in enumerate(result.get("masks", [])):
        if idx not in enabled_indices:
            continue
        polygon = mask_obj.get("polygon", [])
        if len(polygon) >= 3:
            points = [tuple(map(int, point)) for point in polygon]
            draw.polygon(points, outline="red")
    return image
