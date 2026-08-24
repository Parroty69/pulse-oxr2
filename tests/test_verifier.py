from __future__ import annotations

from src.tier3_reasoning.verifier import ActorVerifier


def _mask(label: str, *, area: int = 100, score: float = 0.8) -> dict:
    return {"label": label, "area_px": area, "score": score}


def test_positive_claim_requires_matching_candidate_region() -> None:
    report = {"findings": "A nodule is present in the right lung."}

    annotation = ActorVerifier().verify(report, [])["findings_annotations"][0]

    assert annotation["claim_type"] == "positive_localizable"
    assert annotation["evidence_status"] == "text_only_unverified"
    assert annotation["grounded"] is False
    assert "no matching candidate region" in annotation["warning"]


def test_matching_pathology_links_positive_claim_without_generic_word_matching() -> None:
    report = {"findings": "A pulmonary nodule is present."}
    masks = [_mask("a chest x-ray showing a pulmonary nodule or mass")]

    annotation = ActorVerifier().verify(report, masks)["findings_annotations"][0]

    assert annotation["evidence_status"] == "candidate_region"
    assert annotation["grounded"] is True
    assert annotation["evidence_ids"] == [0]


def test_negative_claim_is_text_only_instead_of_spatial_grounding_failure() -> None:
    report = {"findings": "There is no evidence of pneumothorax, pleural effusion, or pulmonary edema."}

    annotation = ActorVerifier().verify(report, [])["findings_annotations"][0]

    assert annotation["claim_type"] == "negative_class"
    assert annotation["evidence_status"] == "text_only_negative"
    assert annotation["grounded"] is None
    assert "absence cannot be confirmed" in annotation["warning"]


def test_global_statement_does_not_require_a_lesion_mask() -> None:
    report = {"findings": "A chest x-ray is present."}

    annotation = ActorVerifier().verify(report, [])["findings_annotations"][0]

    assert annotation["claim_type"] == "global_or_other"
    assert annotation["evidence_status"] == "not_applicable"
    assert annotation["warning"] is None


def test_postposed_absence_is_classified_as_negative() -> None:
    report = {"findings": "Pleural effusion is not seen."}

    annotation = ActorVerifier().verify(report, [])["findings_annotations"][0]

    assert annotation["claim_type"] == "negative_class"
    assert annotation["evidence_status"] == "text_only_negative"


def test_preposed_not_is_classified_as_negative() -> None:
    report = {"findings": "There is not a pulmonary nodule."}

    annotation = ActorVerifier().verify(report, [])["findings_annotations"][0]

    assert annotation["claim_type"] == "negative_class"
    assert annotation["evidence_status"] == "text_only_negative"


def test_mixed_draft_only_warns_for_unlinked_positive_finding() -> None:
    report = {
        "findings": (
            "A chest x-ray is present. "
            "The lung fields are clear, and there is no evidence of pneumothorax, pleural effusion, or "
            "pulmonary edema. A nodule is present in the lung field."
        )
    }

    annotations = ActorVerifier().verify(report, [])["findings_annotations"]

    assert [annotation["evidence_status"] for annotation in annotations] == [
        "not_applicable",
        "text_only_negative",
        "text_only_unverified",
    ]
    assert sum(annotation["grounded"] is False for annotation in annotations) == 1
