from __future__ import annotations

import json
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None


class ReportGenerator:
    """Swappable backend: CheXagent-2-3b (CausalLM) or MedGemma-4B-it (image-text-to-text pipeline)."""

    def __init__(self, backend: str = "chexagent", device: str = "cuda", use_mock: bool = False):
        self.backend = backend
        self.use_mock = use_mock
        self.tokenizer = None
        self.model = None
        self.pipe = None

        if use_mock or torch is None:
            self.backend = "mock"
            return

        if backend == "chexagent" and AutoTokenizer and AutoModelForCausalLM:
            model_name = "StanfordAIMI/CheXagent-2-3b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", trust_remote_code=True
            ).to(torch.bfloat16).eval()
        elif backend == "medgemma" and pipeline:
            self.pipe = pipeline(
                "image-text-to-text",
                model="google/medgemma-4b-it",
                torch_dtype=torch.bfloat16,
                device=device,
            )
        else:
            self.backend = "mock"

    def _build_context(self, tier1_scores: list[dict], tier2_masks: list[dict]) -> str:
        top_scores = sorted(tier1_scores, key=lambda s: s.get("score", 0.0), reverse=True)[:8]
        score_lines = [f"- {s.get('label')}: {s.get('score', 0.0):.3f} @ {s.get('full_image_bbox')}" for s in top_scores]
        mask_lines = [
            f"- {m.get('label') or 'region'} score={m.get('score', 0.0):.3f} box={m.get('box')} area_px={m.get('area_px', 0)}"
            for m in tier2_masks
        ]
        return "\n".join(
            [
                "Tier 1 candidate findings:",
                *(score_lines or ["- none"]),
                "Tier 2 grounded masks:",
                *(mask_lines or ["- none"]),
            ]
        )

    def _mock_generate(self, tier1_scores: list[dict], tier2_masks: list[dict]) -> dict:
        findings = []
        if tier2_masks:
            for m in sorted(tier2_masks, key=lambda x: x.get("score", 0.0), reverse=True)[:3]:
                findings.append(
                    f"Possible {m.get('label', 'abnormality')} with spatial grounding at box {m.get('box')} (area {m.get('area_px', 0)} px)."
                )
        else:
            findings.append("No spatially grounded focal abnormality was detected by the current model thresholds.")

        impression = (
            "AI-assisted draft for clinician review: correlate with full clinical context and physician interpretation."
        )
        return {
            "findings": " ".join(findings),
            "impression": impression,
            "draft_status": "AI-Assisted Draft — Requires Physician Review",
            "grounding_context": self._build_context(tier1_scores, tier2_masks),
        }

    def generate_report(self, global_thumbnail, tier1_scores: list[dict], tier2_masks: list[dict]) -> dict:
        if self.backend == "mock" or (self.model is None and self.pipe is None):
            return self._mock_generate(tier1_scores, tier2_masks)

        grounding = self._build_context(tier1_scores, tier2_masks)
        prompt = (
            "You are an expert radiology assistant drafting a report for physician review only; "
            "do not produce a final diagnosis. Output sections FINDINGS and IMPRESSION.\n\n"
            f"Grounding context:\n{grounding}\n"
        )

        if self.backend == "chexagent" and self.model is not None and self.tokenizer is not None:
            tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**tokens, max_new_tokens=180)
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        elif self.backend == "medgemma" and self.pipe is not None:
            result = self.pipe({"image": global_thumbnail, "text": prompt}, max_new_tokens=180)
            text = result[0].get("generated_text", "")
        else:
            return self._mock_generate(tier1_scores, tier2_masks)

        findings, impression = _parse_sections(text)
        return {
            "findings": findings,
            "impression": impression,
            "draft_status": "AI-Assisted Draft — Requires Physician Review",
            "grounding_context": grounding,
        }


def _parse_sections(text: str) -> tuple[str, str]:
    upper = text.upper()
    f_idx = upper.find("FINDINGS")
    i_idx = upper.find("IMPRESSION")
    if f_idx == -1 and i_idx == -1:
        return text.strip(), ""
    if i_idx == -1:
        findings = text[f_idx:].split(":", 1)[-1].strip()
        return findings, ""
    findings_block = text[f_idx:i_idx] if f_idx != -1 else ""
    impression_block = text[i_idx:]
    findings = findings_block.split(":", 1)[-1].strip()
    impression = impression_block.split(":", 1)[-1].strip()
    return findings, impression
