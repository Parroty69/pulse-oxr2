from __future__ import annotations

import asyncio
import gc
import time
from pathlib import Path
from typing import Any

import yaml

from src.ingestion.dicom_loader import load_dicom, to_rgb
from src.ingestion.tiling import build_dual_stream
from src.tier1_screening.biomedclip_screener import BiomedCLIPScreener
from src.tier2_grounding.medsam_segmenter import MedSAMSegmenter
from src.tier3_reasoning.report_generator import ReportGenerator
from src.tier3_reasoning.verifier import ActorVerifier


SCORE_THRESHOLD = 0.5
NON_LOCALIZABLE_LABEL_TERMS = ("normal", "no finding", "no acute", "clear lungs")


def _iou(a: list[int], b: list[int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def _union_box(a: list[int], b: list[int]) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _is_localizable_label(label: str) -> bool:
    normalized = label.lower().strip()
    return bool(normalized) and not any(term in normalized for term in NON_LOCALIZABLE_LABEL_TERMS)


def tier1_scores_to_medsam_boxes(
    tile_scores: list[dict],
    threshold: float = SCORE_THRESHOLD,
    *,
    merge_iou: float = 0.3,
    max_regions_per_label: int = 2,
) -> list[dict]:
    accepted = [
        t
        for t in tile_scores
        if t.get("score", 0.0) >= threshold
        and _is_localizable_label(str(t.get("label", "")))
        and len(t.get("full_image_bbox", [])) == 4
    ]
    grouped: dict[str, list[dict]] = {}
    for row in accepted:
        grouped.setdefault(str(row.get("label", "unknown")), []).append(row)

    output: list[dict] = []
    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)
        merged: list[dict] = []
        for row in rows:
            box = list(row.get("full_image_bbox", [0, 0, 0, 0]))
            attached = False
            for m in merged:
                if _iou(box, m["box_xyxy"]) > merge_iou:
                    m["box_xyxy"] = _union_box(m["box_xyxy"], box)
                    m["score"] = max(m["score"], float(row.get("score", 0.0)))
                    attached = True
                    break
            if not attached:
                merged.append(
                    {
                        "label": label,
                        "score": float(row.get("score", 0.0)),
                        "box_xyxy": box,
                    }
                )
        output.extend(merged[:max_regions_per_label])

    return output


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively apply a generated runtime overlay without discarding user config."""

    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _release_accelerator_memory(device: str) -> None:
    gc.collect()
    try:
        import torch
    except Exception:  # pragma: no cover
        return
    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif (
            device.startswith("mps")
            and hasattr(torch, "mps")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.empty_cache()
        elif device.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except (RuntimeError, AttributeError):
        pass


async def run_pipeline(dicom_path: str, config: dict) -> dict:
    pipeline_started = time.perf_counter()
    runtime_metrics: dict[str, float] = {}
    pipeline_cfg = config.get("pipeline", {})
    model_cfg = config.get("model", {})

    tile_size = int(pipeline_cfg.get("tile_size", 1024))
    overlap = float(pipeline_cfg.get("tile_overlap", 0.2))
    threshold = float(pipeline_cfg.get("score_threshold", SCORE_THRESHOLD))
    batch_size = int(pipeline_cfg.get("tile_batch_size", 4))
    max_concurrency = int(pipeline_cfg.get("max_gpu_concurrency", 1))

    phi_tags = config.get("phi_tags", [])
    vocab = config.get("vocab_prompts", ["a normal chest x-ray"])

    stage_started = time.perf_counter()
    load_result = load_dicom(dicom_path, phi_tags=phi_tags)
    full_image = load_result.image_np
    global_thumbnail, tiles, _ = build_dual_stream(full_image, tile_size=tile_size, overlap=overlap)
    runtime_metrics["ingestion_ms"] = round((time.perf_counter() - stage_started) * 1000, 1)

    use_mock = bool(model_cfg.get("use_mock_models", False))
    device = model_cfg.get("device", "cpu")

    verifier = ActorVerifier(min_area_px=int(pipeline_cfg.get("mask_min_area_px", 64)))
    release_between_tiers = bool(model_cfg.get("release_between_tiers", True))

    sem = asyncio.Semaphore(max_concurrency)
    screener = BiomedCLIPScreener(
        device=device,
        dtype=model_cfg.get("screening_dtype"),
        use_mock=use_mock,
    )

    async def score_batch(batch_tiles: list[Any]) -> list[dict]:
        async with sem:
            serializable = [
                {"tile_index": t.tile_index, "image": t.image, "full_image_bbox": t.full_image_bbox}
                for t in batch_tiles
            ]
            return await asyncio.to_thread(screener.score_tiles, serializable, vocab)

    stage_started = time.perf_counter()
    batches = [tiles[i : i + batch_size] for i in range(0, len(tiles), batch_size)]
    scored_batches = await asyncio.gather(*(score_batch(batch) for batch in batches))
    tile_scores = [row for batch in scored_batches for row in batch]
    runtime_metrics["screening_ms"] = round((time.perf_counter() - stage_started) * 1000, 1)
    if release_between_tiers:
        del screener
        _release_accelerator_memory(device)

    medsam_prompts = tier1_scores_to_medsam_boxes(
        tile_scores,
        threshold=threshold,
        merge_iou=float(pipeline_cfg.get("box_merge_iou", 0.3)),
        max_regions_per_label=int(pipeline_cfg.get("max_regions_per_label", 2)),
    )

    stage_started = time.perf_counter()
    image_rgb = to_rgb(full_image)
    if medsam_prompts:
        segmenter = MedSAMSegmenter(
            checkpoint_path=model_cfg.get("medsam_checkpoint_path", "medsam_vit_b.pth"),
            device=device,
            use_mock=use_mock,
        )
        masks = await asyncio.to_thread(segmenter.segment_from_boxes, image_rgb, medsam_prompts)
        if release_between_tiers:
            del segmenter
            _release_accelerator_memory(device)
    else:
        masks = []
    runtime_metrics["grounding_ms"] = round((time.perf_counter() - stage_started) * 1000, 1)

    stage_started = time.perf_counter()
    reporter = ReportGenerator(
        backend=model_cfg.get("tier3_backend", "chexagent"),
        device=device,
        use_mock=use_mock,
        runtime=model_cfg.get("tier3_runtime", "transformers"),
        quantization=model_cfg.get("tier3_quantization", "none"),
        compute_dtype=model_cfg.get("compute_dtype", "bf16"),
        model_id=model_cfg.get("tier3_model_id"),
    )
    report = await asyncio.to_thread(reporter.generate_report, global_thumbnail, tile_scores, masks)
    resolved_tier3_model_id = reporter.model_id
    if release_between_tiers:
        del reporter
        _release_accelerator_memory(device)
    verified_report = verifier.verify(report, masks)
    runtime_metrics["reasoning_ms"] = round((time.perf_counter() - stage_started) * 1000, 1)
    runtime_metrics["total_ms"] = round((time.perf_counter() - pipeline_started) * 1000, 1)

    warnings = list(load_result.warnings)
    if use_mock:
        warnings.append("Mock-model mode is active; outputs are synthetic and are not medical inference.")
    else:
        warnings.append(
            "BiomedCLIP scores are relative to the configured prompt set and are not calibrated clinical "
            "probabilities."
        )
    if not masks:
        warnings.append(
            "No candidate-region overlay was produced. This does not mean the image is normal; "
            "positive report statements remain text-only unless linked to an overlay."
        )

    return {
        "display_image": image_rgb,
        "masks": masks,
        "report": verified_report,
        "tile_scores": tile_scores,
        "warnings": warnings,
        "runtime_metrics": runtime_metrics,
        "model_manifest": {
            "screening_model_id": BiomedCLIPScreener.MODEL_ID,
            "tier3_backend": model_cfg.get("tier3_backend", "chexagent"),
            "tier3_runtime": model_cfg.get("tier3_runtime", "transformers"),
            "tier3_model_id": resolved_tier3_model_id,
            "tier3_quantization": model_cfg.get("tier3_quantization", "none"),
            "device": device,
            "mock_models": use_mock,
            "tile_size": tile_size,
            "score_threshold": threshold,
        },
    }
