from __future__ import annotations

from io import BytesIO
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "ui" / "app.py"


def _dicom_bytes(pixel_value: int) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "OT"
    dataset.Rows = 256
    dataset.Columns = 256
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.full((256, 256), pixel_value, dtype=np.uint16).tobytes()
    stream = BytesIO()
    pydicom.dcmwrite(stream, dataset, enforce_file_format=True)
    return stream.getvalue()


def test_ui_imports_from_outside_repository(tmp_path: Path) -> None:
    test_script = textwrap.dedent(
        f"""
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file({str(APP_PATH)!r}).run(timeout=10)
        if app.exception:
            raise RuntimeError("; ".join(exception.message for exception in app.exception))
        """
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", test_script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_changing_upload_clears_previous_analysis_result() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    model_select = next(widget for widget in app.selectbox if widget.label == "Reasoning model")
    model_select.set_value("mock").run(timeout=30)
    assert any("cpu · mock · none" in caption.value for caption in app.caption)
    app.file_uploader[0].upload("first.dcm", _dicom_bytes(10), "application/dicom").run(timeout=30)
    app.button[0].click().run(timeout=30)

    assert app.session_state["result"] is not None

    app.file_uploader[0].upload("second.dcm", _dicom_bytes(20), "application/dicom").run(timeout=30)

    assert app.session_state["result"] is None

    app.selectbox[0].set_value("Tiếng Việt").run(timeout=30)

    assert app.subheader[0].value == "Trình xem DICOM"
