# CXR Co-Pilot (Research Prototype)

⚠️ **AI-Assisted Draft — Requires Physician Review.**

This repository implements a research-grade, multi-agent chest X-ray co-pilot pipeline:
- Tier 1: BiomedCLIP open-vocabulary screening
- Tier 2: MedSAM spatial grounding masks
- Tier 3: CheXagent-2-3b or MedGemma-4B draft report generation

## Clinical and regulatory disclaimers
- This is a **clinical decision-support co-pilot**, not an autonomous diagnostic system.
- Outputs are **drafts** that require licensed clinician review, editing, and sign-off.
- Model artifacts (BiomedCLIP, MedSAM, CheXagent, MedGemma) are research artifacts and not FDA-cleared devices.
- The current CheXagent-2-3b Hugging Face model card lists an **MIT** license; verify the model card and all bundled components for your deployment.
- MedGemma usage is governed by the **Health AI Developer Foundations** license (check current terms before deployment).

## PHI handling
DICOM ingestion strips configured identifying tags before image tensors are processed. The app avoids logging PHI and never logs raw pixel data.

## One-click hardware-aware deployment

On macOS or Linux:

```bash
./deploy.sh
```

On Windows PowerShell (CUDA, Intel XPU, or CPU):

```bash
.\deploy.ps1
```

The deployer detects NVIDIA CUDA, AMD ROCm, Apple Silicon, Intel XPU/SYCL, or CPU; measures currently available accelerator memory; installs the matching PyTorch wheel; selects the Tier-3 runtime and quantization; verifies the installed accelerator; downloads the selected checkpoints; writes `config/runtime.generated.yaml`; and launches Streamlit.

Apple Silicon uses PyTorch MPS for BiomedCLIP/MedSAM and MLX-VLM for MedGemma. NVIDIA, AMD, and Intel use Transformers with native CUDA, ROCm, or XPU and memory-aware bitsandbytes quantization. BiomedCLIP and MedSAM remain floating-point because post-training low-bit conversion is not validated for this pipeline.

Preview the complete decision without changing the machine:

```bash
./deploy.sh --dry-run --no-launch
```

MedGemma requires accepting Google's terms on its Hugging Face model page and setting `HF_TOKEN`. Use `./deploy.sh --backend chexagent` if MedGemma is not available to the deployment account. The deployer intentionally does not install kernel GPU drivers; it checks them and gives an actionable failure before model download.

See [Hardware acceleration and quantization](docs/ACCELERATION.md) for the support matrix, memory policy, overrides, and driver prerequisites.

## Manual development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/ui/app.py
```

The manual path uses the mock configuration until a runtime overlay is generated or supplied through `CXR_PIPELINE_CONFIG`.

## Tests
```bash
pytest tests/test_tiling.py tests/test_pipeline_integration.py
```
