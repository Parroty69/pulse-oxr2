from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from src.tier1_screening.biomedclip_screener import BiomedCLIPScreener


class _FakeModel:
    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.eye(tokens.shape[0], dtype=torch.float32)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.ones((images.shape[0], 2), dtype=torch.float32)


def test_score_tiles_converts_normalized_numpy_input_to_pil_rgb() -> None:
    screener = BiomedCLIPScreener(device="cpu", dtype="fp32", use_mock=True)
    screener.model = _FakeModel()
    screener.tokenizer = lambda prompts: torch.zeros((len(prompts), 1), dtype=torch.int64)
    received: list[Image.Image] = []

    def preprocess(image: Image.Image) -> torch.Tensor:
        received.append(image)
        return torch.zeros((3, 224, 224), dtype=torch.float32)

    screener.preprocess = preprocess
    scores = screener.score_tiles(
        [
            {
                "image": np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8),
                "full_image_bbox": [0, 0, 8, 8],
            }
        ],
        ["finding", "normal"],
    )

    assert len(received) == 1
    assert received[0].mode == "RGB"
    assert received[0].size == (8, 8)
    assert np.asarray(received[0]).min() == 0
    assert np.asarray(received[0]).max() == 255
    assert len(scores) == 2
    assert sum(row["score"] for row in scores) == pytest.approx(1.0)
