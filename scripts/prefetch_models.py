#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError


BIOMEDCLIP_MODEL = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch the models selected by the deployment planner")
    parser.add_argument("--tier3-model", required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN") or None

    try:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
