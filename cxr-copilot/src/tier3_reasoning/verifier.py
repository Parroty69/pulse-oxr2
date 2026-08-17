from __future__ import annotations

import re


class ActorVerifier:
    """Ensures every sentence in the draft report is grounded in a Tier 2 mask."""

    def __init__(self, min_area_px: int = 64, min_score: float = 0.0):
        self.min_area_px = min_area_px
        self.min_score = min_score

    def _sentences(self, text: str) -> list[str]:
        chunks = re.split(r"(?<=[.!?])\s+", text.strip())
        return [c.strip() for c in chunks if c.strip()]

    def verify(self, report: dict, tier2_masks: list[dict]) -> dict:
        findings = report.get("findings", "")
        sentences = self._sentences(findings)

        registry: list[tuple[str, dict]] = []
        for m in tier2_masks:
            if m.get("area_px", 0) < self.min_area_px or m.get("score", 0.0) < self.min_score:
                continue
            label = str(m.get("label") or "").lower().strip()
            if label:
                registry.append((label, m))

        annotations = []
        for sent in sentences:
            s_lower = sent.lower()
            grounded = False
            for label, _ in registry:
                token_pool = [label, *label.replace("-", " ").split(), *label.split()]
                if any(token and token in s_lower for token in token_pool):
                    grounded = True
                    break
            annotations.append(
                {
                    "sentence": sent,
                    "grounded": grounded,
                    "warning": None if grounded else "Unverified — no spatial grounding found",
                }
            )

        verified = dict(report)
        verified["findings_annotations"] = annotations
        return verified
