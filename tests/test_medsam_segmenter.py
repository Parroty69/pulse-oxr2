from __future__ import annotations

import pytest

from src.tier2_grounding import medsam_segmenter


class _FakeModel:
    def __init__(self):
        self.state_dict = None
        self.device = None
        self.evaluating = False

    def load_state_dict(self, state_dict) -> None:
        self.state_dict = state_dict

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        self.evaluating = True
        return self


def test_real_mode_fails_closed_when_segment_anything_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(medsam_segmenter, "sam_model_registry", None)

    with pytest.raises(RuntimeError, match="segment-anything"):
        medsam_segmenter.MedSAMSegmenter("checkpoint.pth", device="cpu")


def test_checkpoint_is_safely_loaded_on_cpu_before_model_moves_to_accelerator(monkeypatch) -> None:
    model = _FakeModel()
    load_calls = []
    predictor = object()
    monkeypatch.setattr(medsam_segmenter, "sam_model_registry", {"vit_b": lambda: model})
    monkeypatch.setattr(medsam_segmenter, "SamPredictor", lambda loaded_model: predictor)

    def fake_load(path, *, map_location, weights_only):
        load_calls.append((path, map_location, weights_only))
        return {"encoder.weight": "test-value"}

    monkeypatch.setattr(medsam_segmenter.torch, "load", fake_load)

    segmenter = medsam_segmenter.MedSAMSegmenter("checkpoint.pth", device="mps")

    assert load_calls == [("checkpoint.pth", "cpu", True)]
    assert model.state_dict == {"encoder.weight": "test-value"}
    assert model.device == "mps"
    assert model.evaluating is True
    assert segmenter.predictor is predictor
