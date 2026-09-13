#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VENV_PYTHON = ROOT / ".venv/bin/python"
if sys.prefix == sys.base_prefix and VENV_PYTHON.is_file():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cxr-copilot-matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from src.deployment.hardware import detect_machine, select_deployment_plan
from src.ingestion.dicom_loader import load_dicom
from src.orchestration.pipeline import deep_merge, load_yaml


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an offline software-readiness check for the Pulse-OXR research prototype."
    )
    parser.add_argument("--dicom", type=Path, help="Optional de-identified DICOM to decode without model inference")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the pytest suite")
    return parser


def main() -> int:
    args = _parser().parse_args()
    results: list[tuple[str, str, str]] = []

    def record(status: str, check: str, detail: str) -> None:
        results.append((status, check, detail))

    version = sys.version_info
    if version >= (3, 10):
        record("PASS", "Python", f"{version.major}.{version.minor}.{version.micro}")
    else:
        record("FAIL", "Python", "Python 3.10 or newer is required")

    modules = (
        "numpy",
        "pydicom",
        "torch",
        "streamlit",
        "yaml",
        "PIL",
        "open_clip",
        "segment_anything",
        "transformers",
    )
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        record("FAIL", "Python packages", f"Missing: {', '.join(missing)}")
    else:
        record("PASS", "Python packages", f"{len(modules)} required modules found")

    base_config_path = ROOT / "config/pipeline_config.yaml"
    runtime_path = ROOT / "config/runtime.generated.yaml"
    config = load_yaml(base_config_path)
    if runtime_path.is_file():
        config = deep_merge(config, load_yaml(runtime_path))
        record("PASS", "Runtime config", "Generated hardware selection is present")
    else:
        record("WARN", "Runtime config", "No generated runtime config; repository mock defaults will be used")

    model = config.get("model", {})
    mock_models = bool(model.get("use_mock_models", True))
    runtime = str(model.get("tier3_runtime", "mock"))
    runtime_module = {"mlx-vlm": "mlx_vlm", "transformers": "transformers"}.get(runtime)
    if runtime_module and importlib.util.find_spec(runtime_module) is None:
        record("FAIL", "Selected runtime", f"Python module {runtime_module} is missing")
    else:
        record(
            "PASS",
            "Selected runtime",
            f"{model.get('tier3_backend', 'mock')} via {runtime} ({model.get('tier3_quantization', 'none')})",
        )

    checkpoint = Path(model.get("medsam_checkpoint_path", "medsam_vit_b.pth"))
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if mock_models:
        record("PASS", "Model mode", "Mock mode selected; output must be presented as synthetic")
    elif checkpoint.is_file() and checkpoint.stat().st_size > 300 * 1024 * 1024:
        record("PASS", "MedSAM checkpoint", f"Present ({checkpoint.stat().st_size / 1024**2:.0f} MiB)")
    else:
        record("FAIL", "MedSAM checkpoint", "Missing or incomplete checkpoint")

    model_id = model.get("tier3_model_id")
    if not mock_models and model_id:
        try:
            from huggingface_hub import try_to_load_from_cache

            cached_config = try_to_load_from_cache(str(model_id), "config.json")
            if isinstance(cached_config, str):
                record("PASS", "Tier-3 cache", f"{model_id} configuration found locally")
            else:
                record("WARN", "Tier-3 cache", f"Could not confirm a local config for {model_id}")
        except (OSError, RuntimeError, ValueError) as exc:
            record("WARN", "Tier-3 cache", f"Cache inspection failed: {exc}")

    try:
        profile = detect_machine()
        plan = select_deployment_plan(profile)
        record(
            "PASS" if profile.runtime_ready else "WARN",
            "Hardware detection",
            f"{profile.device_name}; {plan.framework}; {plan.quantization}; {profile.available_memory_gib:.1f} GiB available",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        record("FAIL", "Hardware detection", str(exc))

    if args.dicom:
        try:
            phi = load_yaml(ROOT / "config/phi_tags.yaml").get("phi_tags", [])
            decoded = load_dicom(args.dicom, phi_tags=phi)
            height, width = decoded.image_np.shape
            record("PASS", "DICOM decode", f"{width}x{height}; {len(decoded.stripped_tags)} configured PHI fields cleared in memory")
            for warning in decoded.warnings:
                record("WARN", "DICOM notice", warning)
        except (OSError, RuntimeError, ValueError) as exc:
            record("FAIL", "DICOM decode", str(exc))

    if not args.skip_tests:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode == 0:
            record("PASS", "Automated tests", "pytest suite passed")
        else:
            record("FAIL", "Automated tests", f"pytest exited with status {completed.returncode}")
    else:
        record("WARN", "Automated tests", "Skipped by request")

    print("\nPulse-OXR software readiness preflight")
    print("Software readiness only - not clinical validation or diagnostic-performance evidence.\n")
    for status, check, detail in results:
        print(f"[{status:4}] {check}: {detail}")

    failures = sum(status == "FAIL" for status, _, _ in results)
    warnings = sum(status == "WARN" for status, _, _ in results)
    print(f"\nSummary: {failures} failure(s), {warnings} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
