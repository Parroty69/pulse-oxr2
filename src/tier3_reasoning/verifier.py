from __future__ import annotations

import re


PATHOLOGY_TERMS: dict[str, tuple[str, ...]] = {
    "pneumothorax": ("pneumothorax",),
    "pleural_effusion": ("pleural effusion", "effusion"),
    "pulmonary_edema": ("pulmonary edema", "edema"),
    "pulmonary_nodule_or_mass": (
        "pulmonary nodule",
        "lung nodule",
        "micro-nodule",
        "micronodule",
        "nodule",
        "lung mass",
        "pulmonary mass",
        "mass",
    ),
}

NEGATION_PATTERN = re.compile(
    r"(?:no evidence of|negative for|absence of|free of|without|not|no|absent)\b[^.!?;]{0,90}$"
)
POST_NEGATION_PATTERN = re.compile(
    r"^\s*(?:(?:is|are|was|were)\s+)?(?:absent|not present|not seen|not identified|not detected)\b"
)


def _pathology_key(text: str) -> str | None:
    normalized = text.lower()
    for key, terms in PATHOLOGY_TERMS.items():
        if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms):
            return key
    return None


def _mentioned_pathologies(text: str) -> list[str]:
    normalized = text.lower()
    return [
        key
        for key, terms in PATHOLOGY_TERMS.items()
        if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in terms)
    ]


def _is_negated(text: str, pathology: str) -> bool:
    terms = PATHOLOGY_TERMS[pathology]
    clauses = re.split(r"\b(?:but|however)\b|;", text.lower())
    for clause in clauses:
        for term in terms:
            match = re.search(rf"\b{re.escape(term)}\b", clause)
            if match:
                if NEGATION_PATTERN.search(clause[: match.start()]):
                    return True
                if POST_NEGATION_PATTERN.search(clause[match.end() :]):
                    return True
    return False


class ActorVerifier:
    """Classify report claims and link positive localizable claims to candidate regions."""

    def __init__(self, min_area_px: int = 64, min_score: float = 0.0):
        self.min_area_px = min_area_px
        self.min_score = min_score

    def _sentences(self, text: str) -> list[str]:
        chunks = re.split(r"(?<=[.!?])\s+", text.strip())
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def verify(self, report: dict, tier2_masks: list[dict]) -> dict:
        registry: dict[str, list[int]] = {}
        for index, mask in enumerate(tier2_masks):
            if mask.get("area_px", 0) < self.min_area_px or mask.get("score", 0.0) < self.min_score:
                continue
            pathology = _pathology_key(str(mask.get("label") or ""))
            if pathology:
                registry.setdefault(pathology, []).append(index)

        annotations = []
        for sentence in self._sentences(report.get("findings", "")):
            mentioned = _mentioned_pathologies(sentence)
            negative = [key for key in mentioned if _is_negated(sentence, key)]
            positive = [key for key in mentioned if key not in negative]

            if positive:
                evidence_ids = sorted({index for key in positive for index in registry.get(key, [])})
                missing = [key for key in positive if key not in registry]
                fully_linked = not missing
                status = "candidate_region" if fully_linked else "text_only_unverified"
                warning = None
                if missing:
                    labels = ", ".join(key.replace("_", " ") for key in missing)
                    warning = f"Text-only positive claim; no matching candidate region for: {labels}."
                claim_type = "positive_localizable"
                grounded: bool | None = fully_linked
            elif negative:
                evidence_ids = []
                status = "text_only_negative"
                warning = (
                    "Negative statement is text-only; absence cannot be confirmed by a candidate-region overlay."
                )
                claim_type = "negative_class"
                grounded = None
            else:
                evidence_ids = []
                status = "not_applicable"
                warning = None
                claim_type = "global_or_other"
                grounded = None

            annotations.append(
                {
                    "sentence": sentence,
                    "claim_type": claim_type,
                    "labels": positive or negative,
                    "evidence_ids": evidence_ids,
                    "evidence_status": status,
                    "grounded": grounded,
                    "warning": warning,
                }
            )

        verified = dict(report)
        verified["findings_annotations"] = annotations
        return verified
