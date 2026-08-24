from __future__ import annotations

import math

import numpy as np
from PIL import Image
import pytest
import torch

from src.tier1_screening import biomedclip_screener
from src.tier1_screening.biomedclip_screener import BiomedCLIPScreener


class _FakeModel:
    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.eye(tokens.shape[0], dtype=torch.float32)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.ones((images.shape[0], 2), dtype=torch.float32)


class _ScaledFakeModel(_FakeModel):
    logit_scale = torch.tensor(math.log(10.0), dtype=torch.float32)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[1.0, 0.0]] * images.shape[0], dtype=torch.float32)


def test_real_mode_fails_closed_when_open_clip_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(biomedclip_screener, "create_model_from_pretrained", None)

    with pytest.raises(RuntimeError, match="open_clip_torch"):
        BiomedCLIPScreener(device="cpu", dtype="fp32", use_mock=False)


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


def test_score_tiles_applies_learned_logit_scale_and_preserves_tile_index() -> None:
    screener = BiomedCLIPScreener(device="cpu", dtype="fp32", use_mock=True)
    screener.model = _ScaledFakeModel()
    screener.tokenizer = lambda prompts: torch.zeros((len(prompts), 1), dtype=torch.int64)
    screener.preprocess = lambda image: torch.zeros((3, 224, 224), dtype=torch.float32)

    scores = screener.score_tiles(
        [
            {
                "tile_index": 17,
                "image": np.zeros((8, 8), dtype=np.float32),
                "full_image_bbox": [10, 20, 30, 40],
            }
        ],
        ["finding", "normal"],
    )

    assert scores[0]["tile_index"] == 17
    assert scores[0]["score"] > 0.99
    assert scores[1]["score"] < 0.01
