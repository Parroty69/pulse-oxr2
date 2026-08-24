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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cxr-copilot-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import streamlit as st

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.ingestion.dicom_loader import load_dicom, to_rgb
from src.orchestration.pipeline import deep_merge, load_yaml, run_pipeline
from src.ui.dicom_viewer import render_overlay


CONFIG_DIR = ROOT / "config"
RUNTIME_CONFIG = Path(os.environ.get("CXR_PIPELINE_CONFIG", CONFIG_DIR / "runtime.generated.yaml"))
FEEDBACK_LOG = Path("/tmp/cxr_copilot_feedback.log")
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

UI_TEXT = {
    "English": {
        "disclaimer": (
            "> ⚠️ **Research prototype.** AI-assisted draft only - requires licensed physician review. "
            "Not a diagnostic device."
        ),
        "language": "Interface language / Ngôn ngữ",
        "upload": "Upload DICOM",
        "model": "Reasoning model",
        "run": "Run analysis",
        "runtime": "Local runtime",
        "how_it_works": "How the evidence pipeline works",
        "screening_stage": "1. Screening",
        "screening_desc": "BiomedCLIP ranks four target findings across image tiles.",
        "region_stage": "2. Candidate regions",
        "region_desc": "MedSAM converts selected tiles into experimental overlays.",
        "draft_stage": "3. Draft + evidence audit",
        "draft_desc": "MedGemma/CheXagent drafts text; the verifier labels its evidence status.",
        "too_large": "DICOM exceeds the 32 MiB safety limit.",
        "spinner": "Running screening, candidate-region generation, and report drafting...",
        "failed": "Analysis failed safely: {error}",
        "viewer": "DICOM Viewer",
        "overlay": "Show experimental candidate-region overlay",
        "overlay_caption": "Candidate regions are AI proposals, not validated lesion boundaries.",
        "per_finding": "Per-region overlay toggles",
        "no_regions": "No candidate-region overlay was generated for this run.",
        "preview_recovery": "DICOM preview recovery failed: {error}",
        "preview_error": "DICOM preview unavailable: {error}. Please run the analysis again.",
        "empty_viewer": "Upload one DICOM and run analysis to view the radiograph.",
        "notices": "Safety and input notices",
        "report": "Draft Radiology Report",
        "report_language": "Model-generated report text is currently produced in English.",
        "positive_unlinked": "Text-only positive claim - no matching candidate-region evidence.",
        "evidence_audit": "Claim evidence audit",
        "candidate_region": "label-linked to experimental candidate region",
        "text_only_unverified": "text-only positive claim",
        "text_only_negative": "text-only negative claim; absence is not spatially verifiable",
        "not_applicable": "global/non-localizable statement",
        "findings": "Findings",
        "impression": "Impression",
        "record_edit": "Record report edit feedback",
        "edit_recorded": "Edit counts and a one-way hash were recorded locally; report text and PHI were not stored.",
        "empty_report": "Upload one DICOM and run analysis to see a draft report.",
        "runtime_details": "Runtime and reproducibility details",
        "ingestion": "Ingestion",
        "screening": "Screening",
        "grounding": "Candidate regions",
        "reasoning": "Reasoning",
        "total": "Total",
    },
    "Tiếng Việt": {
        "disclaimer": (
            "> ⚠️ **Nguyên mẫu nghiên cứu.** Bản nháp do AI hỗ trợ - bắt buộc bác sĩ có giấy phép xem xét. "
            "Không phải thiết bị chẩn đoán."
        ),
        "language": "Ngôn ngữ / Interface language",
        "upload": "Tải ảnh DICOM",
        "model": "Mô hình suy luận",
        "run": "Chạy phân tích",
        "runtime": "Cấu hình xử lý cục bộ",
        "how_it_works": "Luồng xử lý bằng chứng",
        "screening_stage": "1. Sàng lọc",
        "screening_desc": "BiomedCLIP xếp hạng bốn dấu hiệu mục tiêu trên các ô ảnh.",
        "region_stage": "2. Vùng ứng viên",
        "region_desc": "MedSAM chuyển các ô được chọn thành lớp phủ thử nghiệm.",
        "draft_stage": "3. Bản nháp + kiểm tra bằng chứng",
        "draft_desc": "MedGemma/CheXagent soạn văn bản; bộ kiểm tra gắn trạng thái bằng chứng.",
        "too_large": "Tệp DICOM vượt quá giới hạn an toàn 32 MiB.",
        "spinner": "Đang sàng lọc, tạo vùng ứng viên và soạn báo cáo...",
        "failed": "Phân tích đã dừng an toàn: {error}",
        "viewer": "Trình xem DICOM",
        "overlay": "Hiện lớp phủ vùng ứng viên thử nghiệm",
        "overlay_caption": "Vùng ứng viên là đề xuất của AI, không phải ranh giới tổn thương đã được xác thực.",
        "per_finding": "Bật/tắt từng vùng ứng viên",
        "no_regions": "Lần chạy này không tạo được vùng ứng viên.",
        "preview_recovery": "Không thể khôi phục bản xem trước DICOM: {error}",
        "preview_error": "Không thể hiển thị DICOM: {error}. Vui lòng chạy lại phân tích.",
        "empty_viewer": "Tải một tệp DICOM và chạy phân tích để xem ảnh.",
        "notices": "Thông báo an toàn và dữ liệu đầu vào",
        "report": "Bản nháp báo cáo X-quang",
        "report_language": "Nội dung báo cáo do mô hình tạo hiện được xuất bằng tiếng Anh.",
        "positive_unlinked": "Nhận định dương tính chỉ có văn bản - không có vùng ứng viên tương ứng.",
        "evidence_audit": "Kiểm tra bằng chứng cho từng nhận định",
        "candidate_region": "được liên kết theo nhãn với vùng ứng viên thử nghiệm",
        "text_only_unverified": "nhận định dương tính chỉ có văn bản",
        "text_only_negative": "nhận định âm tính chỉ có văn bản; không thể xác minh sự vắng mặt bằng lớp phủ",
        "not_applicable": "nhận định toàn cục/không thể định vị",
        "findings": "Mô tả",
        "impression": "Kết luận",
        "record_edit": "Ghi nhận phản hồi chỉnh sửa",
        "edit_recorded": "Chỉ số lần sửa và mã băm một chiều đã được lưu cục bộ; nội dung báo cáo và PHI không được lưu.",
        "empty_report": "Tải một tệp DICOM và chạy phân tích để xem bản nháp.",
        "runtime_details": "Chi tiết thời gian chạy và khả năng tái lập",
        "ingestion": "Nạp dữ liệu",
        "screening": "Sàng lọc",
        "grounding": "Vùng ứng viên",
        "reasoning": "Suy luận",
        "total": "Tổng cộng",
    },
}


def _configured_pipeline() -> dict:
    config = load_yaml(CONFIG_DIR / "pipeline_config.yaml")
    if RUNTIME_CONFIG.is_file():
        config = deep_merge(config, load_yaml(RUNTIME_CONFIG))
    return config


def _decode_uploaded_preview(uploaded_bytes: bytes):
    """Decode a preview independently of the pipeline result contract."""

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
    with FEEDBACK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


st.set_page_config(page_title="Pulse-OXR CXR Co-Pilot", layout="wide")

if "result" not in st.session_state:
    st.session_state.result = None
if "upload_digest" not in st.session_state:
    st.session_state.upload_digest = None
if "selected_backend" not in st.session_state:
    st.session_state.selected_backend = None

configured_pipeline = _configured_pipeline()
configured_model = configured_pipeline.get("model", {})

with st.sidebar:
    language = st.selectbox(UI_TEXT["English"]["language"], list(UI_TEXT))
    text = UI_TEXT[language]
    uploaded = st.file_uploader(text["upload"], type=["dcm", "dicom"])
    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    upload_digest = hashlib.sha256(uploaded_bytes).hexdigest() if uploaded_bytes is not None else None
    if upload_digest != st.session_state.upload_digest:
        st.session_state.result = None
        st.session_state.upload_digest = upload_digest

    upload_too_large = uploaded_bytes is not None and len(uploaded_bytes) > MAX_UPLOAD_BYTES
    if upload_too_large:
        st.error(text["too_large"])

    configured_backend = configured_model.get("tier3_backend", "mock")
    if configured_model.get("use_mock_models", True):
        backend_options = ["mock"]
    else:
        backend_options = [configured_backend, "mock"]
    backend = st.selectbox(text["model"], list(dict.fromkeys(backend_options)))
    if backend != st.session_state.selected_backend:
        st.session_state.result = None
        st.session_state.selected_backend = backend

    displayed_model = (
        {
            "device": "cpu",
            "tier3_runtime": "mock",
            "tier3_quantization": "none",
        }
        if backend == "mock"
        else configured_model
    )
    st.caption(
        f"{text['runtime']}: {displayed_model.get('device', 'cpu')} · "
        f"{displayed_model.get('tier3_runtime', 'mock')} · "
        f"{displayed_model.get('tier3_quantization', 'none')}"
    )
    run_btn = st.button(
        text["run"],
        disabled=uploaded_bytes is None or upload_too_large,
        type="primary",
    )

st.markdown(text["disclaimer"])
with st.expander(text["how_it_works"]):
    stage_columns = st.columns(3)
    stage_columns[0].markdown(f"**{text['screening_stage']}**\n\n{text['screening_desc']}")
    stage_columns[1].markdown(f"**{text['region_stage']}**\n\n{text['region_desc']}")
    stage_columns[2].markdown(f"**{text['draft_stage']}**\n\n{text['draft_desc']}")

if run_btn and uploaded_bytes is not None:
    analysis_error = None
    with st.spinner(text["spinner"]):
        with NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(uploaded_bytes)
            dicom_path = tmp.name

        try:
            pipeline_config = _configured_pipeline()
            vocab = load_yaml(CONFIG_DIR / "pathology_vocab.yaml")
            phi = load_yaml(CONFIG_DIR / "phi_tags.yaml")
            model_config = pipeline_config.setdefault("model", {})
            model_config["tier3_backend"] = backend
            model_config["use_mock_models"] = backend == "mock"
            if backend == "mock":
                model_config.update(
                    {
                        "device": "cpu",
                        "tier3_runtime": "mock",
                        "tier3_quantization": "none",
                        "compute_dtype": "fp32",
                        "screening_dtype": "fp32",
                    }
                )
            merged_config = {
                **pipeline_config,
                "vocab_prompts": vocab.get("vocab_prompts", []),
                "phi_tags": phi.get("phi_tags", []),
            }
            pipeline_result = asyncio.run(run_pipeline(dicom_path, merged_config))
            if pipeline_result.get("display_image") is None:
                pipeline_result["display_image"] = _decode_uploaded_preview(uploaded_bytes)
            st.session_state.result = pipeline_result
        except Exception as exc:  # pragma: no cover - surfaced safely in the UI
            st.session_state.result = None
            analysis_error = f"{type(exc).__name__}: {exc}"
        finally:
            Path(dicom_path).unlink(missing_ok=True)

    if analysis_error:
        st.error(text["failed"].format(error=analysis_error))

col_viewer, col_report = st.columns([2, 1])
result = st.session_state.result

with col_viewer:
    st.subheader(text["viewer"])
    show_masks = st.toggle(text["overlay"], value=True)
    st.caption(text["overlay_caption"])

    if result:
        preview_recovery_error = None
        if result.get("display_image") is None and uploaded_bytes is not None:
            try:
                result["display_image"] = _decode_uploaded_preview(uploaded_bytes)
            except Exception as exc:  # pragma: no cover - surfaced in the UI
                preview_recovery_error = exc

        if preview_recovery_error is not None:
            st.error(text["preview_recovery"].format(error=preview_recovery_error))
        else:
            masks = result.get("masks", [])
            enabled: set[int] = set()
            if masks:
                st.caption(text["per_finding"])
                for index, mask_obj in enumerate(masks):
                    label = mask_obj.get("label") or f"Region {index + 1}"
                    is_on = st.checkbox(f"{label} ({index})", value=True, key=f"mask_{index}")
                    if is_on:
                        enabled.add(index)
            else:
                st.info(text["no_regions"])

            try:
                preview = render_overlay(result, enabled if show_masks else set())
            except (TypeError, ValueError) as exc:
                st.error(text["preview_error"].format(error=exc))
            else:
                st.image(preview, use_container_width=True)

        notices = result.get("warnings", [])
        if notices:
            with st.expander(text["notices"]):
                for notice in notices:
                    st.warning(notice)
    else:
        st.info(text["empty_viewer"])

with col_report:
    st.subheader(text["report"])
    if result:
        report = result["report"]
        st.caption(text["report_language"])
        annotations = report.get("findings_annotations", [])
        for annotation in annotations:
            if annotation.get("evidence_status") == "text_only_unverified":
                st.warning(f"{annotation.get('sentence')} - {text['positive_unlinked']}")

        if annotations:
            with st.expander(text["evidence_audit"]):
                icons = {
                    "candidate_region": "🟢",
                    "text_only_unverified": "🟠",
                    "text_only_negative": "🔵",
                    "not_applicable": "⚪",
                }
                for annotation in annotations:
                    status = annotation.get("evidence_status", "not_applicable")
                    st.markdown(
                        f"{icons.get(status, '⚪')} **{text.get(status, status)}:** "
                        f"{annotation.get('sentence', '')}"
                    )

        findings_text = st.text_area(text["findings"], value=report.get("findings", ""), height=250)
        st.text_area(text["impression"], value=report.get("impression", ""), height=150)
        if st.button(text["record_edit"]):
            _log_feedback(report.get("findings", ""), findings_text)
            st.success(text["edit_recorded"])

        metrics = result.get("runtime_metrics", {})
        manifest = result.get("model_manifest", {})
        with st.expander(text["runtime_details"]):
            metric_keys = ["ingestion_ms", "screening_ms", "grounding_ms", "reasoning_ms", "total_ms"]
            metric_labels = [
                text["ingestion"],
                text["screening"],
                text["grounding"],
                text["reasoning"],
                text["total"],
            ]
            columns = st.columns(len(metric_keys))
            for column, key, label in zip(columns, metric_keys, metric_labels):
                column.metric(label, f"{metrics.get(key, 0.0):.0f} ms")
            st.json(manifest)
    else:
        st.info(text["empty_report"])

st.markdown(
    "---\n"
    "AI-Assisted Draft - Requires Physician Review. "
    "BiomedCLIP, MedSAM, CheXagent, and MedGemma are research models and not cleared diagnostic devices."
)
