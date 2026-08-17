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
- CheXagent-2-3b is distributed under **CC-BY-NC-4.0** (non-commercial).
- MedGemma usage is governed by the **Health AI Developer Foundations** license (check current terms before deployment).

## PHI handling
DICOM ingestion strips configured identifying tags before image tensors are processed. The app avoids logging PHI and never logs raw pixel data.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run UI
```bash
./scripts/run_ui.sh
```

The launcher resolves the repository root automatically, so it works even when
invoked from another directory. Additional Streamlit flags are passed through,
for example: `./scripts/run_ui.sh --server.port 8502`.

UI footer also includes the non-diagnostic disclaimer.

## Tests
```bash
pytest tests/
```
