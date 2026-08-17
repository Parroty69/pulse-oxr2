# Repository Guidelines

## Project Structure & Module Organization

Application code lives under `src/`. DICOM loading and tiling are in
`src/ingestion/`; model adapters are grouped by pipeline stage in
`src/tier1_screening/`, `src/tier2_grounding/`, and `src/tier3_reasoning/`.
`src/orchestration/pipeline.py` coordinates those stages, while
`src/ui/app.py` provides the Streamlit interface. Runtime YAML files belong in
`config/`, operational helpers in `scripts/`, and automated checks in `tests/`.
Keep model checkpoints and test DICOMs out of version control.

## Build, Test, and Development Commands

Create and populate the local environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `./scripts/run_ui.sh` to start the UI using the repository's virtual
environment. Extra Streamlit options pass through, for example
`./scripts/run_ui.sh --server.port 8502`. Run `pytest tests/` for the complete
test suite, or target a file during development, such as
`pytest tests/test_tiling.py -q`. Use `bash -n scripts/*.sh` after editing shell
scripts.

## Coding Style & Naming Conventions

Use four-space indentation and conventional PEP 8 naming: `snake_case` for
functions and variables, `CapWords` for classes, and `UPPER_CASE` for constants.
Retain type annotations on public functions and dataclasses. Keep model-specific
logic behind its tier adapter and avoid coupling the UI directly to model
implementations. No formatter or linter is currently enforced, so keep imports
organized and run `git diff --check` before committing.

## Testing Guidelines

Tests use `pytest` and `pytest-asyncio`; name files `test_*.py` and test functions
`test_*`. Prefer small deterministic fixtures and mock-model mode so tests do not
download checkpoints or require a GPU. Add regression coverage for changes to
DICOM handling, coordinate mapping, pipeline payloads, or UI startup. There is
no formal coverage threshold, but every bug fix should include a focused test.

## Commit & Pull Request Guidelines

History uses short, imperative subjects such as `Add portable Streamlit UI
launcher` and `Implement CXR copilot pipeline scaffold`. Keep commits focused
and avoid mixing model assets with source changes. Pull requests should explain
the behavior changed, list verification commands and results, and call out model
licenses or configuration changes. Include screenshots for visible UI changes
and link the relevant issue when one exists.

## Security & Clinical Safety

Never commit PHI, real patient DICOMs, credentials, or downloaded model weights.
Preserve the clinician-review disclaimer and PHI-stripping behavior. Treat all
generated reports as research drafts, not diagnoses.
