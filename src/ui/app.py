from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.ingestion.dicom_loader import load_dicom, to_rgb
from src.orchestration.pipeline import deep_merge, load_yaml, run_pipeline
from src.ui.dicom_viewer import render_overlay


CONFIG_DIR = ROOT / "config"
RUNTIME_CONFIG = Path(os.environ.get("CXR_PIPELINE_CONFIG", CONFIG_DIR / "runtime.generated.yaml"))
FEEDBACK_LOG = Path("/tmp/cxr_copilot_feedback.log")


def _configured_pipeline() -> dict:
    config = load_yaml(CONFIG_DIR / "pipeline_config.yaml")
    if RUNTIME_CONFIG.is_file():
        config = deep_merge(config, load_yaml(RUNTIME_CONFIG))
    return config


def _decode_uploaded_preview(uploaded_bytes: bytes):
    """Decode a preview independently of the pipeline result contract.

    Keeping this fallback in the Streamlit entrypoint also covers a running
    server that hot-reloaded the UI while retaining an older imported pipeline
    module in memory.
    """

    with NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(uploaded_bytes)
        dicom_path = tmp.name
    try:
        return to_rgb(load_dicom(dicom_path, phi_tags=[]).image_np)
    finally:
        Path(dicom_path).unlink(missing_ok=True)


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
    configured_model = _configured_pipeline().get("model", {})
    configured_backend = configured_model.get("tier3_backend", "mock")
    if configured_model.get("use_mock_models", True):
        backend_options = ["mock"]
    else:
        backend_options = [configured_backend, "mock"]
    backend = st.selectbox("Reasoning model", list(dict.fromkeys(backend_options)))
    run_btn = st.button("Run Analysis", disabled=uploaded is None)

if "result" not in st.session_state:
    st.session_state.result = None

if run_btn and uploaded:
    with st.spinner("Running Tier 1 → Tier 2 → Tier 3 pipeline..."):
        with NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            dicom_path = tmp.name

        pipeline_config = _configured_pipeline()
        vocab = load_yaml(CONFIG_DIR / "pathology_vocab.yaml")
        phi = load_yaml(CONFIG_DIR / "phi_tags.yaml")
        pipeline_config["model"]["tier3_backend"] = backend
        pipeline_config["model"]["use_mock_models"] = backend == "mock"

        merged_config = {
            **pipeline_config,
            "vocab_prompts": vocab.get("vocab_prompts", []),
            "phi_tags": phi.get("phi_tags", []),
        }

        try:
            result = asyncio.run(run_pipeline(dicom_path, merged_config))
            if result.get("display_image") is None:
                result["display_image"] = _decode_uploaded_preview(uploaded.getvalue())
        finally:
            Path(dicom_path).unlink(missing_ok=True)
        st.session_state.result = result

with col_viewer:
    st.subheader("DICOM Viewer")
    result = st.session_state.result
    show_masks = st.toggle("Show MedSAM overlay", value=True)

    if result:
        preview_recovery_error = None
        if result.get("display_image") is None and uploaded is not None:
            try:
                result["display_image"] = _decode_uploaded_preview(uploaded.getvalue())
            except Exception as exc:  # pragma: no cover - surfaced in the UI
                preview_recovery_error = exc

        if preview_recovery_error is not None:
            st.error(f"DICOM preview recovery failed: {preview_recovery_error}")
        else:
            st.caption("Per-finding overlay toggles")
            enabled = set()
            for idx, mask_obj in enumerate(result.get("masks", [])):
                label = mask_obj.get("label") or f"Region {idx + 1}"
                is_on = st.checkbox(f"{label} ({idx})", value=True, key=f"mask_{idx}")
                if is_on:
                    enabled.add(idx)

            try:
                preview = render_overlay(result, enabled if show_masks else set())
            except (TypeError, ValueError) as exc:
                st.error(f"DICOM preview unavailable: {exc}. Please run the analysis again.")
            else:
                st.image(preview, use_container_width=True)
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
