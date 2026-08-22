from __future__ import annotations

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from segment_anything import SamPredictor, sam_model_registry
except Exception:  # pragma: no cover
    SamPredictor = None
    sam_model_registry = None


class MedSAMSegmenter:
    def __init__(self, checkpoint_path: str, device: str = "cuda", use_mock: bool = False):
        self.device = device
        self.use_mock = use_mock
        self.predictor = None
        if not use_mock and sam_model_registry is not None and SamPredictor is not None:
            if torch is None:
                raise RuntimeError("PyTorch is required to load MedSAM")
            model = sam_model_registry["vit_b"]()
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(device).eval()
            self.predictor = SamPredictor(model)

    def _rectangle_polygon(self, box: list[int]) -> list[list[int]]:
        x0, y0, x1, y1 = [int(v) for v in box]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    def segment_from_boxes(self, image_np, boxes_xyxy: list[list[int]]) -> list[dict]:
        h, w = image_np.shape[:2]
        results: list[dict] = []

        if self.predictor is not None:
            rgb = image_np
            if image_np.ndim == 2:
                rgb = np.stack([image_np, image_np, image_np], axis=-1)
            self.predictor.set_image(rgb)

        for item in boxes_xyxy:
            box = item["box_xyxy"] if isinstance(item, dict) and "box_xyxy" in item else item
            x0, y0, x1, y1 = [int(v) for v in box]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)

            if self.predictor is not None:
                masks, scores, _ = self.predictor.predict(
                    box=np.array([x0, y0, x1, y1]),
                    multimask_output=False,
                )
                mask = masks[0].astype(bool)
            else:
                mask = np.zeros((h, w), dtype=bool)
                if x1 > x0 and y1 > y0:
                    mask[y0:y1, x0:x1] = True

            polygon = self._rectangle_polygon([x0, y0, x1, y1])
            if cv2 is not None:
                cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    largest = max(cnts, key=cv2.contourArea)
                    polygon = largest.reshape(-1, 2).astype(int).tolist()

            results.append(
                {
                    "label": item.get("label") if isinstance(item, dict) else None,
                    "score": float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
                    "box": [x0, y0, x1, y1],
                    "mask": mask,
                    "polygon": polygon,
                    "area_px": int(mask.sum()),
                }
            )
        return results
