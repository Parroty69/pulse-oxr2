from __future__ import annotations

import numpy as np
import pytest

from src.ui.dicom_viewer import prepare_display_image, render_overlay


def test_prepare_display_image_scales_normalized_grayscale() -> None:
    grayscale = np.array([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32)

    display = prepare_display_image(grayscale)

    assert display.shape == (2, 2, 3)
    assert display.dtype == np.uint8
    assert np.array_equal(display[..., 0], display[..., 1])
    assert display[0, 0, 0] == 0
    assert display[1, 1, 0] == 255


def test_render_overlay_preserves_radiograph_without_enabled_masks() -> None:
    base = np.arange(64, dtype=np.uint8).reshape(8, 8)
    result = {
        "display_image": base,
        "masks": [{"polygon": [[1, 1], [6, 1], [6, 6], [1, 6]]}],
    }

    rendered = np.asarray(render_overlay(result, set()))

    assert np.array_equal(rendered[..., 0], base)
    assert rendered.max() > 0


def test_render_overlay_draws_enabled_polygon() -> None:
    result = {
        "display_image": np.full((8, 8, 3), 64, dtype=np.uint8),
        "masks": [{"polygon": [[1, 1], [6, 1], [6, 6], [1, 6]]}],
    }

    rendered = np.asarray(render_overlay(result, {0}))

    assert np.any(np.all(rendered == np.array([255, 0, 0]), axis=-1))
    assert np.array_equal(rendered[3, 3], np.array([64, 64, 64]))


def test_prepare_display_image_rejects_missing_image() -> None:
    with pytest.raises(ValueError, match="missing"):
        prepare_display_image(None)
