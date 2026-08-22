from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from open_clip import create_model_from_pretrained, get_tokenizer
except Exception:  # pragma: no cover
    create_model_from_pretrained = None
    get_tokenizer = None


class BiomedCLIPScreener:
    MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self, device: str = "cuda", dtype: Any = None, use_mock: bool = False):
        self.device = device
        self.use_mock = use_mock
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.dtype = self._resolve_dtype(dtype, device)

        if not use_mock and torch is not None and create_model_from_pretrained is not None:
            self.model, self.preprocess = create_model_from_pretrained(self.MODEL_ID)
            self.model = self.model.to(device, dtype=self.dtype).eval()
            self.tokenizer = get_tokenizer(self.MODEL_ID)

    @staticmethod
    def _resolve_dtype(dtype: Any, device: str):
        if torch is None:
            return None
        if dtype is None:
            return torch.float32 if device == "cpu" else torch.float16
        if isinstance(dtype, str):
            normalized = dtype.lower().replace("torch.", "")
            mapping = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            if normalized not in mapping:
                raise ValueError(f"Unsupported BiomedCLIP dtype: {dtype}")
            return mapping[normalized]
        return dtype

    def _prepare_image(self, tile: Any):
        if isinstance(tile, dict):
            tile = tile.get("image", tile)
        if isinstance(tile, np.ndarray):
            return tile
        if hasattr(tile, "image"):
            return np.asarray(tile.image)
        return np.asarray(tile)

    def _tile_bbox(self, tile: Any) -> list[int]:
        if isinstance(tile, dict) and "full_image_bbox" in tile:
            return list(tile["full_image_bbox"])
        if hasattr(tile, "full_image_bbox"):
            return list(tile.full_image_bbox)
        return [0, 0, 0, 0]

    def _mock_scores(self, tiles: list[Any], vocab_prompts: list[str]) -> list[dict]:
        outputs: list[dict] = []
        for idx, tile in enumerate(tiles):
            arr = self._prepare_image(tile).astype(np.float32)
            intensity = float(arr.mean()) if arr.size else 0.0
            logits = []
            for prompt in vocab_prompts:
                p = prompt.lower()
                if "normal" in p:
                    logits.append(1.0 - intensity)
                else:
                    logits.append(intensity)
            logits_np = np.array(logits, dtype=np.float32)
            exp = np.exp(logits_np - logits_np.max())
            probs = exp / max(exp.sum(), 1e-8)
            bbox = self._tile_bbox(tile)
            for label, score in zip(vocab_prompts, probs.tolist()):
                outputs.append(
                    {
                        "tile_index": idx,
                        "label": label,
                        "score": float(score),
                        "full_image_bbox": bbox,
                    }
                )
        return outputs

    def score_tiles(self, tiles: list, vocab_prompts: list[str]) -> list[dict]:
        if self.model is None or torch is None:
            return self._mock_scores(tiles, vocab_prompts)

        with torch.no_grad():
            text_tokens = self.tokenizer(vocab_prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            outputs: list[dict] = []
            processed = []
            for tile in tiles:
                arr = self._prepare_image(tile)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                processed.append(self.preprocess(arr))
            image_tensor = torch.stack(processed).to(self.device, dtype=self.dtype)
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

            for idx, row in enumerate(probs):
                bbox = self._tile_bbox(tiles[idx])
                for label, score in zip(vocab_prompts, row.tolist()):
                    outputs.append(
                        {
                            "tile_index": idx,
                            "label": label,
                            "score": float(score),
                            "full_image_bbox": bbox,
                        }
                    )
            return outputs
