#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deployment.hardware import (  # noqa: E402
    DeploymentPlan,
    MachineProfile,
    detect_machine,
    force_accelerator,
    profile_from_json,
    pytorch_index_url,
    select_deployment_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the machine, choose an acceleration/quantization profile, install the matching runtime, "
            "download checkpoints, and launch the CXR co-pilot."
        )
    )
    parser.add_argument("--backend", choices=["auto", "medgemma", "chexagent", "mock"], default="auto")
    parser.add_argument("--framework", choices=["auto", "transformers", "pytorch", "mlx", "mlx-vlm"], default="auto")
    parser.add_argument(
        "--quantization",
        choices=["auto", "none", "bf16", "fp16", "int8", "8bit", "6bit", "4bit", "nf4"],
        default="auto",
    )
    parser.add_argument("--accelerator", choices=["auto", "cuda", "rocm", "xpu", "mps", "mlx", "cpu"], default="auto")
    parser.add_argument("--python", default=sys.executable, help="Base Python used to create the virtual environment")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("--config-out", type=Path, default=ROOT / "config/runtime.generated.yaml")
    parser.add_argument("--profile-json", help="Offline hardware profile JSON (file path or literal JSON)")
    parser.add_argument("--detect-only", action="store_true", help="Print the detected profile and selected plan, then exit")
    parser.add_argument("--dry-run", action="store_true", help="Print every decision and command without changing files")
    parser.add_argument("--skip-install", action="store_true", help="Reuse an already prepared virtual environment")
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip checkpoint prefetching; MedSAM must already exist and Hugging Face models load lazily",
    )
    parser.add_argument("--no-launch", action="store_true", help="Prepare the machine without starting Streamlit")
    parser.add_argument(
        "--allow-unverified-runtime",
        action="store_true",
        help="Continue when an AMD/Intel system driver could not be verified (the PyTorch probe must still pass)",
    )
    return parser


def _display_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def _run(command: list[str], *, dry_run: bool, env: dict[str, str] | None = None) -> None:
    print(f"  $ {_display_command(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _validate_base_python(python: str, dry_run: bool) -> None:
    if dry_run:
        return
    output = subprocess.run(
        [python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    major, minor = (int(part) for part in output.split(".")[:2])
    if (major, minor) < (3, 10):
        raise RuntimeError(f"Python 3.10 or newer is required; found {output}")


def _install_environment(
    profile: MachineProfile,
    plan: DeploymentPlan,
    venv: Path,
    base_python: str,
    dry_run: bool,
) -> Path:
    venv_python = _venv_python(venv)
    if not venv_python.exists():
        _run([base_python, "-m", "venv", str(venv)], dry_run=dry_run)
    _run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        dry_run=dry_run,
    )

    torch_command = [str(venv_python), "-m", "pip", "install", "--upgrade", "torch", "torchvision"]
    index_url = pytorch_index_url(profile)
    if index_url:
        torch_command.extend(["--index-url", index_url])
    _run(torch_command, dry_run=dry_run)
    _run(
        [str(venv_python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
        dry_run=dry_run,
    )

    extras: list[str] = []
    if plan.framework == "mlx-vlm":
        extras.append("mlx-vlm")
    if plan.framework == "transformers" and plan.quantization in {"int8", "nf4"}:
        extras.append("bitsandbytes")
    if extras:
        _run([str(venv_python), "-m", "pip", "install", "--upgrade", *extras], dry_run=dry_run)
    return venv_python


def _prefetch_models(venv_python: Path, plan: DeploymentPlan, dry_run: bool) -> None:
    if plan.use_mock_models:
        print("  Mock deployment selected; no checkpoints are required.")
        return
    checkpoint = ROOT / "checkpoints/medsam/medsam_vit_b.pth"
    command = [
        str(venv_python),
        str(ROOT / "scripts/prefetch_models.py"),
        "--medsam-checkpoint",
        str(checkpoint),
        "--tier3-model",
        plan.model_id,
    ]
    _run(command, dry_run=dry_run)


def _runtime_overlay(profile: MachineProfile, plan: DeploymentPlan) -> dict[str, object]:
    return {
        "model": {
            "device": plan.torch_device,
            "tier3_backend": plan.backend,
            "tier3_runtime": plan.framework,
            "tier3_model_id": plan.model_id,
            "tier3_quantization": plan.quantization,
            "compute_dtype": plan.torch_dtype,
            "screening_dtype": plan.screening_dtype,
            "medsam_checkpoint_path": str((ROOT / "checkpoints/medsam/medsam_vit_b.pth").resolve()),
            "use_mock_models": plan.use_mock_models,
            "release_between_tiers": plan.release_between_tiers,
        },
        "pipeline": {
            "tile_batch_size": plan.tile_batch_size,
            "max_gpu_concurrency": 1,
        },
        "deployment": {
            "hardware": profile.to_dict(),
            "selection": plan.to_dict(),
        },
    }


def _write_runtime_config(path: Path, profile: MachineProfile, plan: DeploymentPlan, dry_run: bool) -> None:
    payload = json.dumps(_runtime_overlay(profile, plan), indent=2) + "\n"
    print(f"  Write runtime overlay -> {path}")
    if dry_run:
        print(payload)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _probe_runtime(venv_python: Path, profile: MachineProfile, plan: DeploymentPlan, dry_run: bool) -> None:
    command = [
        str(venv_python),
        str(ROOT / "scripts/probe_acceleration.py"),
        "--hardware-backend",
        profile.accelerator,
        "--framework",
        plan.framework,
        "--quantization",
        plan.quantization,
        "--device-index",
        str(profile.device_index),
    ]
    _run(command, dry_run=dry_run)


def _print_selection(profile: MachineProfile, plan: DeploymentPlan) -> None:
    memory_label = "unified memory budget" if profile.unified_memory else "available VRAM"
    print("\nDetected machine")
    print(f"  OS:           {profile.os_name}/{profile.architecture}")
    print(f"  Accelerator:  {profile.accelerator} ({profile.device_name})")
    print(f"  Memory:       {profile.available_memory_gib:.1f} GiB {memory_label}")
    print(f"  Torch device: {profile.torch_device}")
    print("\nSelected deployment")
    print(f"  Tier-3 model: {plan.backend} ({plan.model_id or 'mock'})")
    print(f"  Framework:    {plan.framework}")
    print(f"  Quantization: {plan.quantization}")
    print(f"  Compute type: {plan.torch_dtype}")
    print(f"  Tile batch:   {plan.tile_batch_size}")
    for warning in plan.warnings:
        print(f"  Warning:      {warning}")


def _launch(venv_python: Path, config_path: Path, profile: MachineProfile, dry_run: bool) -> None:
    env = os.environ.copy()
    env["CXR_PIPELINE_CONFIG"] = str(config_path.resolve())
    if profile.accelerator == "mps":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    command = [str(venv_python), "-m", "streamlit", "run", str(ROOT / "src/ui/app.py")]
    _run(command, dry_run=dry_run, env=env)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.profile_json:
        profile = profile_from_json(args.profile_json)
    else:
        preferred = None if args.accelerator == "auto" else args.accelerator
        profile = detect_machine(preferred)
    if args.profile_json and args.accelerator != "auto":
        profile = force_accelerator(profile, args.accelerator)
    plan = select_deployment_plan(profile, args.backend, args.framework, args.quantization)
    _print_selection(profile, plan)

    if args.detect_only:
        print("\nJSON")
        print(json.dumps({"hardware": profile.to_dict(), "selection": plan.to_dict()}, indent=2))
        return 0
    if not profile.runtime_ready and not (
        args.allow_unverified_runtime or args.dry_run or args.skip_install
    ):
        raise RuntimeError("The selected hardware runtime is not ready; resolve the warning above or use CPU explicitly")

    _validate_base_python(args.python, args.dry_run)
    if args.skip_install:
        venv_python = _venv_python(args.venv)
        if not args.dry_run and not venv_python.exists():
            raise FileNotFoundError(f"Virtual environment Python not found: {venv_python}")
    else:
        print("\nPreparing Python environment")
        venv_python = _install_environment(profile, plan, args.venv, args.python, args.dry_run)

    print("\nVerifying acceleration runtime")
    _probe_runtime(venv_python, profile, plan, args.dry_run)

    if not args.skip_model_download:
        print("\nPrefetching selected checkpoints")
        _prefetch_models(venv_python, plan, args.dry_run)

    print("\nGenerating runtime configuration")
    _write_runtime_config(args.config_out, profile, plan, args.dry_run)

    if not args.no_launch:
        print("\nLaunching CXR Co-Pilot")
        _launch(venv_python, args.config_out, profile, args.dry_run)
    else:
        print(f"\nSetup complete. Launch with: CXR_PIPELINE_CONFIG={args.config_out} {venv_python} -m streamlit run src/ui/app.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"\nDeployment failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
