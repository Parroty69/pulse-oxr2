#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints

echo "Downloading BiomedCLIP metadata..."
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", local_dir="checkpoints/biomedclip", allow_patterns=["*.json", "*.txt", "*.model", "*.bin", "*.safetensors"])
PY

echo "Downloading MedSAM checkpoint metadata..."
echo "Please place medsam_vit_b.pth in checkpoints/medsam/ (or configure path in config/pipeline_config.yaml)."

echo "Downloading CheXagent metadata..."
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="StanfordAIMI/CheXagent-2-3b", local_dir="checkpoints/chexagent", allow_patterns=["*.json", "*.txt", "*.model", "*.bin", "*.safetensors"])
PY

echo "Downloading MedGemma metadata..."
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/medgemma-4b-it", local_dir="checkpoints/medgemma", allow_patterns=["*.json", "*.txt", "*.model", "*.bin", "*.safetensors"])
PY

echo "Done. Ensure you have accepted upstream model licenses before use."
