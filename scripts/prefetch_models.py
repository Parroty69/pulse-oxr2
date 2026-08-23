#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
import requests


BIOMEDCLIP_MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
MEDSAM_URL = "https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1"
MEDSAM_MD5 = "3bb6db55bd0c9ca30b61248bca72f8d6"


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_medsam(destination: Path) -> None:
    if destination.is_file() and _file_md5(destination) == MEDSAM_MD5:
        print(f"  MedSAM checkpoint already verified: {destination}", flush=True)
        return

    print(f"  Downloading MedSAM checkpoint to {destination}...", flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    downloaded_bytes = 0
    next_progress_bytes = 64 * 1024 * 1024
    try:
        with requests.get(
            MEDSAM_URL,
            headers={"User-Agent": "cxr-copilot-deployer/1.0"},
            stream=True,
            timeout=(15, 60),
        ) as response:
            response.raise_for_status()
            with partial.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        downloaded_bytes += len(chunk)
                        if downloaded_bytes >= next_progress_bytes:
                            print(f"    downloaded {downloaded_bytes / 1024**2:.0f} MiB", flush=True)
                            next_progress_bytes += 64 * 1024 * 1024
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    checksum = _file_md5(partial)
    if checksum != MEDSAM_MD5:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"MedSAM checksum mismatch: expected {MEDSAM_MD5}, received {checksum}")
    partial.replace(destination)
    print(f"  MedSAM checkpoint verified ({checksum}).", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch the models selected by the deployment planner")
    parser.add_argument("--medsam-checkpoint", type=Path)
    parser.add_argument("--tier3-model", required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or None

    try:
        if args.medsam_checkpoint:
            download_medsam(args.medsam_checkpoint)
        if "medgemma" in args.tier3_model.lower():
            print("  Verifying licensed access to google/medgemma-4b-it...", flush=True)
            hf_hub_download(repo_id="google/medgemma-4b-it", filename="config.json", token=token)
        for model_id in (BIOMEDCLIP_MODEL, args.tier3_model):
            print(f"  Downloading {model_id} into the Hugging Face cache...", flush=True)
            snapshot_download(repo_id=model_id, token=token)
    except GatedRepoError as exc:
        print(
            "MedGemma access is gated. Accept Google's Health AI Developer Foundations terms on the "
            "model page, then export HF_TOKEN from that Hugging Face account and rerun deployment.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    except HfHubHTTPError as exc:
        print(f"Hugging Face download failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except requests.RequestException as exc:
        print(f"MedSAM download failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except RuntimeError as exc:
        print(f"Model verification failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
