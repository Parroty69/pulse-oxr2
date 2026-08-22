from __future__ import annotations

import hashlib
from pathlib import Path

from scripts import prefetch_models


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self.payload[:3]
        yield self.payload[3:]


def test_download_medsam_streams_and_verifies_checkpoint(tmp_path: Path, monkeypatch) -> None:
    payload = b"verified-checkpoint"
    expected_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    monkeypatch.setattr(prefetch_models, "MEDSAM_MD5", expected_md5)
    monkeypatch.setattr(
        prefetch_models.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(payload),
    )
    destination = tmp_path / "medsam_vit_b.pth"

    prefetch_models.download_medsam(destination)

    assert destination.read_bytes() == payload
    assert not destination.with_suffix(".pth.part").exists()
