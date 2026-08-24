# CXR Co-Pilot (Research Prototype)

⚠️ **AI-Assisted Draft — Requires Physician Review.**

This repository implements a research-grade, multi-stage chest X-ray co-pilot pipeline:
- Tier 1: BiomedCLIP open-vocabulary screening
- Tier 2: MedSAM experimental candidate-region masks
- Tier 3: CheXagent-2-3b or MedGemma-4B draft report generation

The Streamlit interface supports English and Vietnamese labels, clears stale
results when a different DICOM is selected, displays input/runtime warnings,
and distinguishes positive claims linked to candidate regions from text-only
positive claims, negative statements, and non-localizable statements. Candidate
regions are model proposals, not validated lesion boundaries.
BiomedCLIP scores are relative to the configured prompt set and must not be
interpreted as calibrated disease probabilities.

## Clinical and regulatory disclaimers
- This is a **clinical decision-support co-pilot**, not an autonomous diagnostic system.
- Outputs are **drafts** that require licensed clinician review, editing, and sign-off.
- Model artifacts (BiomedCLIP, MedSAM, CheXagent, MedGemma) are research artifacts and not FDA-cleared devices.
- The current CheXagent-2-3b Hugging Face model card lists an **MIT** license; verify the model card and all bundled components for your deployment.
- MedGemma usage is governed by the **Health AI Developer Foundations** license (check current terms before deployment).

## PHI handling
DICOM ingestion strips configured identifying tags before image tensors are processed. The app avoids logging PHI and never logs raw pixel data.

## One-click hardware-aware deployment

The deployer detects Apple Silicon, NVIDIA CUDA, AMD ROCm, Intel XPU, or a CPU
fallback; installs the matching PyTorch runtime; selects a Tier-3 quantization
from currently available accelerator memory; verifies the accelerator; downloads
and checks model artifacts; writes `config/runtime.generated.yaml`; and launches
the UI.

On macOS or Linux:

```bash
./deploy.sh
```

On Windows PowerShell (CUDA, Intel XPU, or CPU):

```powershell
.\deploy.ps1
```

Use `--detect-only` to inspect the proposed hardware/model plan without changing
the machine, or `--no-launch` to prepare the environment without starting
Streamlit:

```bash
./deploy.sh --detect-only
./deploy.sh --no-launch
```

The automatic Tier-3 choice is MedGemma. On Apple Silicon it uses MLX-VLM and a
BF16, 8-bit, 6-bit, or 4-bit conversion according to the available unified-memory
budget. NVIDIA CUDA, AMD ROCm, and Intel oneAPI/PyTorch XPU use Transformers with
full precision, INT8, or NF4 according to available VRAM and runtime support.
CheXagent uses Transformers on every supported platform; on Apple MPS it uses
FP16 because the current low-bit Transformers path is not reliable there. Use
MedGemma/MLX when an Apple machine cannot fit CheXagent in available unified
memory. Vision tiers remain in floating point.

After setup, `./scripts/run_ui.sh` starts the existing macOS/Linux environment
without reinstalling dependencies or changing the selected model. It resolves
the repository root automatically and forwards Streamlit flags, for example:

```bash
./scripts/run_ui.sh --server.port 8502
```

## Switching between CheXagent and MedGemma

Stop the running UI, then rerun deployment with the desired backend. This safely
replaces the ignored generated runtime configuration; it does not modify
`config/pipeline_config.yaml`.

On macOS or Linux:

```bash
# Google MedGemma 4B (MLX-VLM on Apple Silicon, Transformers elsewhere)
./deploy.sh --backend medgemma

# Stanford CheXagent 2 3B (Transformers)
./deploy.sh --backend chexagent
```

On Windows PowerShell:

```powershell
.\deploy.ps1 --backend medgemma
.\deploy.ps1 --backend chexagent
```

MedGemma is gated. Before selecting it, accept the current terms on the official
[Google MedGemma 4B IT model page](https://huggingface.co/google/medgemma-4b-it)
and expose a read-capable Hugging Face token as `HF_TOKEN` through your shell or
secret manager. The deployer verifies access to Google's official repository
before downloading either the official Transformers weights or an MLX community
conversion. Do not commit the token.

You can override automatic quantization when benchmarking, for example
`./deploy.sh --backend medgemma --quantization 4bit`, but an incompatible
framework/platform combination is rejected instead of silently falling back.

## Real-data validation scope

The `images_001.tar.gz` archive distributed with the
[NIH ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC) is useful
for independent image-integrity, ingestion, tiling, and robustness tests. It is
not identified as MedGemma training data in Google's
[instruction-tuned](https://huggingface.co/google/medgemma-4b-it) or
[pretrained](https://huggingface.co/google/medgemma-4b-pt) model cards. Do not
describe it as MedGemma training data.

The image tarball alone has no finding labels or official patient-level split.
Diagnostic metrics such as sensitivity, specificity, or AUROC therefore require
the matching NIH metadata/labels and a leakage-safe evaluation protocol. Passing
archive, preprocessing, or inference smoke tests is not evidence of clinical
accuracy.

UI footer also includes the non-diagnostic disclaimer.

## Deadline preflight and competition demo

Run the offline software-readiness check before a demonstration. It validates the
environment, selected hardware plan, checkpoint presence, optional DICOM decode,
and complete automated test suite; it does not claim clinical validation.

```bash
./scripts/preflight.py --dicom /path/to/de-identified-demo.dcm
```

The submission-ready title, description, timed English narration, Vietnamese
subtitle track, recording checklist, and claim limitations are in
[`docs/competition/DEADLINE_PACKAGE.md`](docs/competition/DEADLINE_PACKAGE.md).

## Tests

```bash
pytest tests/
```
