from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from src.orchestration.pipeline import run_pipeline


def _create_synthetic_dicom(path: Path) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "John^Doe"
    ds.PatientID = "12345"
    ds.Rows = 512
    ds.Columns = 512
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = (np.random.rand(512, 512) * 1024).astype(np.uint16).tobytes()
    ds.save_as(str(path), write_like_original=False)


@pytest.mark.asyncio
async def test_pipeline_result_payload_shape(tmp_path: Path):
    dicom_path = tmp_path / "synthetic.dcm"
    _create_synthetic_dicom(dicom_path)

    config = {
        "model": {
            "device": "cpu",
            "tier3_backend": "mock",
            "use_mock_models": True,
            "medsam_checkpoint_path": "unused.pth",
        },
        "pipeline": {
            "score_threshold": 0.5,
            "tile_size": 256,
            "tile_overlap": 0.2,
            "tile_batch_size": 2,
            "max_gpu_concurrency": 1,
            "mask_min_area_px": 16,
        },
        "phi_tags": ["PatientName", "PatientID"],
        "vocab_prompts": [
            "a chest x-ray showing a pneumothorax",
            "a normal chest x-ray",
        ],
    }

    result = await run_pipeline(str(dicom_path), config)

    assert set(result.keys()) == {"display_image", "masks", "report", "tile_scores", "warnings"}
    assert result["display_image"].shape == (512, 512, 3)
    assert result["display_image"].dtype == np.uint8
    assert result["display_image"].max() > 0
    assert isinstance(result["masks"], list)
    assert isinstance(result["tile_scores"], list)
    assert isinstance(result["warnings"], list)
    assert isinstance(result["report"], dict)
    assert "findings" in result["report"]
    assert "impression" in result["report"]
    assert "findings_annotations" in result["report"]
