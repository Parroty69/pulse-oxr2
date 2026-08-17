from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from src.orchestration.pipeline import load_yaml, run_pipeline


CONFIG_DIR = ROOT / "config"
FEEDBACK_LOG = Path("/tmp/cxr_copilot_feedback.log")


def _render_overlay(result: dict, enabled_indices: set[int]) -> Image.Image:
    base = np.array(result.get("display_image"))
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)
    image = Image.fromarray(base.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for idx, mask_obj in enumerate(result.get("masks", [])):
        if idx not in enabled_indices:
            continue
        poly = mask_obj.get("polygon", [])
        if len(poly) >= 2:
            pts = [tuple(map(int, p)) for p in poly]
            draw.polygon(pts, outline="red")
    return image


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\n", " ").split(".") if part.strip()]


def _log_feedback(original_findings: str, edited_findings: str) -> None:
    original = _split_sentences(original_findings)
    edited = _split_sentences(edited_findings)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "report_edit",
        "original_count": len(original),
        "edited_count": len(edited),
        "deleted_count": max(0, len(original) - len(edited)),
        "edited_hash": hashlib.sha256(edited_findings.encode("utf-8")).hexdigest(),
    }
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


st.set_page_config(page_title="CXR Co-Pilot", layout="wide")
st.markdown(
    "> ⚠️ Research prototype. AI-assisted draft only — requires physician review. "
    "Not a diagnostic device."
)

col_viewer, col_report = st.columns([2, 1])

with st.sidebar:
    uploaded = st.file_uploader("Upload DICOM", type=["dcm"])
    backend = st.selectbox("Reasoning model", ["chexagent", "medgemma", "mock"])
    run_btn = st.button("Run Analysis", disabled=uploaded is None)

if "result" not in st.session_state:
    st.session_state.result = None

if run_btn and uploaded:
    with st.spinner("Running Tier 1 → Tier 2 → Tier 3 pipeline..."):
        with NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(uploaded.read())
            dicom_path = tmp.name

        pipeline_config = load_yaml(CONFIG_DIR / "pipeline_config.yaml")
        vocab = load_yaml(CONFIG_DIR / "pathology_vocab.yaml")
        phi = load_yaml(CONFIG_DIR / "phi_tags.yaml")
        pipeline_config["model"]["tier3_backend"] = backend
        if backend == "mock":
            pipeline_config["model"]["use_mock_models"] = True

        merged_config = {
            **pipeline_config,
            "vocab_prompts": vocab.get("vocab_prompts", []),
            "phi_tags": phi.get("phi_tags", []),
        }

        result = asyncio.run(run_pipeline(dicom_path, merged_config))
        if result.get("masks") and result["masks"][0].get("mask") is not None:
            shape = result["masks"][0]["mask"].shape
            display = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        else:
            display = np.zeros((512, 512, 3), dtype=np.uint8)
        result["display_image"] = display
        st.session_state.result = result

with col_viewer:
    st.subheader("DICOM Viewer")
    result = st.session_state.result
    show_masks = st.toggle("Show MedSAM overlay", value=True)

    if result:
        st.caption("Per-finding overlay toggles")
        enabled = set()
        for idx, mask_obj in enumerate(result.get("masks", [])):
            label = mask_obj.get("label") or f"Region {idx + 1}"
            is_on = st.checkbox(f"{label} ({idx})", value=True, key=f"mask_{idx}")
            if is_on:
                enabled.add(idx)

        if show_masks:
            overlay = _render_overlay(result, enabled)
            st.image(overlay, use_container_width=True)
        else:
            st.image(result.get("display_image"), use_container_width=True)
    else:
        st.info("Upload a DICOM and run analysis to view overlays.")

with col_report:
    st.subheader("Draft Radiology Report")
    if st.session_state.result:
        report = st.session_state.result["report"]
        annotations = report.get("findings_annotations", [])
        for ann in annotations:
            if not ann.get("grounded", True):
                st.warning(f"{ann.get('sentence')} — {ann.get('warning')}")

        findings_text = st.text_area("Findings", value=report.get("findings", ""), height=250)
        impression_text = st.text_area("Impression", value=report.get("impression", ""), height=150)
        if st.button("Save reviewed report"):
            _log_feedback(report.get("findings", ""), findings_text)
            st.success("Reviewed draft saved (local feedback log updated, no PHI stored).")
    else:
        st.info("Upload a DICOM and run analysis to see a draft report here.")

st.markdown(
    "---\n"
    "AI-Assisted Draft — Requires Physician Review. "
    "BiomedCLIP, MedSAM, CheXagent, and MedGemma are research models and not FDA-cleared diagnostic devices."
)
