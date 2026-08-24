from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from src.ingestion.dicom_loader import load_dicom


def _write_dicom(
    path: Path,
    pixels: np.ndarray,
    *,
    photometric: str = "MONOCHROME2",
    modality: str = "DX",
    frames: int = 1,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = modality
    dataset.PatientName = "Example^Person"
    dataset.PatientID = "example-id"
    dataset.Rows = pixels.shape[-2]
    dataset.Columns = pixels.shape[-1]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsStored = 16
    dataset.BitsAllocated = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    if frames > 1:
        dataset.NumberOfFrames = str(frames)
    dataset.PixelData = pixels.astype(np.uint16).tobytes()
    pydicom.dcmwrite(path, dataset, enforce_file_format=True)


def test_monochrome1_is_inverted_and_phi_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / "inverse.dcm"
    _write_dicom(path, np.array([[0, 100], [200, 300]], dtype=np.uint16), photometric="MONOCHROME1")

    result = load_dicom(path, phi_tags=["PatientName", "PatientID"])

    assert result.image_np[0, 0] == pytest.approx(1.0)
    assert result.image_np[1, 1] == pytest.approx(0.0)
    assert result.stripped_tags == ["PatientName", "PatientID"]
    assert any("MONOCHROME1" in warning for warning in result.warnings)


def test_multiframe_dicom_is_rejected_safely(tmp_path: Path) -> None:
    path = tmp_path / "multiframe.dcm"
    _write_dicom(path, np.zeros((2, 4, 4), dtype=np.uint16), frames=2)

    with pytest.raises(ValueError, match="Multi-frame"):
        load_dicom(path)
