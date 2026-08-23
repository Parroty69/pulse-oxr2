from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.deployment import hardware
from src.deployment.hardware import MachineProfile, pytorch_index_url, select_deployment_plan
from src.orchestration.pipeline import deep_merge


ROOT = Path(__file__).resolve().parents[1]


def _profile(
    accelerator: str,
    available: float,
    *,
    total: float | None = None,
    os_name: str = "linux",
    torch_device: str | None = None,
    compute_capability: str | None = None,
    runtime_version: str | None = None,
    unified: bool = False,
) -> MachineProfile:
    devices = {"cuda": "cuda:0", "rocm": "cuda:0", "xpu": "xpu:0", "mps": "mps", "cpu": "cpu"}
    return MachineProfile(
        os_name=os_name,
        architecture="arm64" if os_name == "darwin" else "x86_64",
        accelerator=accelerator,
        device_name={
            "cuda": "NVIDIA RTX 4090",
            "rocm": "AMD Instinct MI300X",
            "xpu": "Intel Arc B580",
            "mps": "Apple M4",
            "cpu": "Test CPU",
        }[accelerator],
        torch_device=torch_device or devices[accelerator],
        total_memory_gib=total or available,
        available_memory_gib=available,
        system_memory_gib=64,
        system_available_memory_gib=32 if accelerator != "cpu" else available,
        unified_memory=unified,
        compute_capability=compute_capability,
        runtime_version=runtime_version,
    )


def test_cuda_high_memory_uses_unquantized_medgemma():
    profile = _profile("cuda", 22, total=24, compute_capability="8.9", runtime_version="13.0")
    plan = select_deployment_plan(profile)

    assert plan.framework == "transformers"
    assert plan.backend == "medgemma"
    assert plan.quantization == "none"
    assert plan.torch_dtype == "bf16"
    assert pytorch_index_url(profile).endswith("/cu130")


@pytest.mark.parametrize(
    ("available", "expected"),
    [(12, "int8"), (8, "int8"), (7.5, "nf4")],
)
def test_cuda_memory_bands_select_transformers_quantization(available: float, expected: str):
    profile = _profile("cuda", available, compute_capability="7.5", runtime_version="12.8")
    plan = select_deployment_plan(profile)

    assert plan.quantization == expected
    assert plan.torch_dtype == "fp16"


def test_pascal_cuda_uses_nf4_instead_of_unsupported_llm_int8():
    profile = _profile("cuda", 8, compute_capability="6.1", runtime_version="12.6")
    plan = select_deployment_plan(profile)

    assert plan.quantization == "nf4"
    assert any("compute capability 7.5" in warning for warning in plan.warnings)


def test_pre_pascal_cuda_recommends_cpu_for_low_bit_loading():
    profile = _profile("cuda", 4, compute_capability="5.2", runtime_version="12.6")

    with pytest.raises(ValueError, match="--accelerator cpu"):
        select_deployment_plan(profile)


def test_rocm_uses_pytorch_cuda_device_semantics():
    profile = _profile("rocm", 12, runtime_version="7.2")
    plan = select_deployment_plan(profile)

    assert plan.hardware_backend == "rocm"
    assert plan.torch_device == "cuda:0"
    assert plan.quantization == "int8"
    assert pytorch_index_url(profile).endswith("/rocm7.2")


def test_intel_xpu_uses_native_pytorch_and_bitsandbytes_policy():
    profile = _profile("xpu", 7, total=12)
    plan = select_deployment_plan(profile)

    assert plan.framework == "transformers"
    assert plan.torch_device == "xpu:0"
    assert plan.quantization == "nf4"
    assert pytorch_index_url(profile).endswith("/xpu")


@pytest.mark.parametrize(
    ("available", "expected_quantization", "expected_model_suffix"),
    [
        (22, "bf16", "-bf16"),
        (16, "8bit", "-8bit"),
        (11, "6bit", "-6bit"),
        (8, "4bit", "-4bit"),
    ],
)
def test_apple_uses_mlx_medgemma_variants(
    available: float, expected_quantization: str, expected_model_suffix: str
):
    profile = _profile("mps", available, total=32, os_name="darwin", unified=True)
    plan = select_deployment_plan(profile)

    assert plan.framework == "mlx-vlm"
    assert plan.quantization == expected_quantization
    assert plan.model_id.endswith(expected_model_suffix)
    assert plan.torch_device == "mps"


def test_apple_chexagent_falls_back_to_transformers_mps():
    profile = _profile("mps", 14, total=24, os_name="darwin", unified=True)
    plan = select_deployment_plan(profile, backend="chexagent")

    assert plan.framework == "transformers"
    assert plan.backend == "chexagent"
    assert plan.quantization == "none"


def test_apple_chexagent_avoids_low_bit_transformers_when_memory_is_tight():
    profile = _profile("mps", 8, total=24, os_name="darwin", unified=True)
    plan = select_deployment_plan(profile, backend="chexagent")

    assert plan.quantization == "none"
    assert any("Low-bit Transformers" in warning for warning in plan.warnings)


def test_explicit_low_bit_transformers_is_rejected_on_apple_mps():
    profile = _profile("mps", 8, total=24, os_name="darwin", unified=True)

    with pytest.raises(ValueError, match="not enabled on Apple MPS"):
        select_deployment_plan(profile, backend="chexagent", quantization="int8")


def test_explicit_mlx_rejects_chexagent_custom_architecture():
    profile = _profile("mps", 14, total=24, os_name="darwin", unified=True)
    with pytest.raises(ValueError, match="CheXagent"):
        select_deployment_plan(profile, backend="chexagent", framework="mlx")


def test_cpu_uses_available_system_memory_and_warns():
    profile = _profile("cpu", 30, total=64)
    plan = select_deployment_plan(profile)

    assert plan.quantization == "int8"
    assert any("slower" in warning for warning in plan.warnings)


def test_quantization_aliases_are_normalized_per_framework():
    cuda = select_deployment_plan(_profile("cuda", 20), quantization="4bit")
    apple = select_deployment_plan(
        _profile("mps", 20, os_name="darwin", unified=True), quantization="int8"
    )

    assert cuda.quantization == "nf4"
    assert apple.quantization == "8bit"


def test_deep_merge_preserves_base_configuration():
    merged = deep_merge(
        {"model": {"device": "cpu", "backend": "mock"}, "pipeline": {"tile_size": 1024}},
        {"model": {"device": "xpu:0"}, "deployment": {"generated": True}},
    )

    assert merged == {
        "model": {"device": "xpu:0", "backend": "mock"},
        "pipeline": {"tile_size": 1024},
        "deployment": {"generated": True},
    }


def test_deployer_dry_run_accepts_offline_profile(tmp_path: Path):
    profile_path = tmp_path / "profile.json"
    config_path = tmp_path / "runtime.yaml"
    profile_path.write_text(
        json.dumps(_profile("xpu", 7, total=12).to_dict()),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/deploy.py"),
            "--profile-json",
            str(profile_path),
            "--dry-run",
            "--no-launch",
            "--skip-model-download",
            "--config-out",
            str(config_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Accelerator:  xpu" in completed.stdout
    assert "Quantization: nf4" in completed.stdout
    assert '"tier3_runtime": "transformers"' in completed.stdout
    assert not config_path.exists()


def test_multi_vendor_detection_prefers_more_free_memory(monkeypatch: pytest.MonkeyPatch):
    cuda = _profile("cuda", 8, total=12)
    rocm = _profile("rocm", 20, total=24)
    monkeypatch.setattr(hardware, "_system_memory", lambda: (64.0, 48.0))
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hardware, "_detect_nvidia", lambda *_: cuda)
    monkeypatch.setattr(hardware, "_detect_amd", lambda *_: rocm)
    monkeypatch.setattr(hardware, "_detect_intel", lambda *_: None)

    assert hardware.detect_machine().accelerator == "rocm"
    assert hardware.detect_machine("cuda").accelerator == "cuda"


def test_multi_vendor_detection_avoids_unready_driver(monkeypatch: pytest.MonkeyPatch):
    cuda = _profile("cuda", 8, total=12)
    rocm = MachineProfile.from_dict({**_profile("rocm", 20, total=24).to_dict(), "runtime_ready": False})
    monkeypatch.setattr(hardware, "_system_memory", lambda: (64.0, 48.0))
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(hardware, "_detect_nvidia", lambda *_: cuda)
    monkeypatch.setattr(hardware, "_detect_amd", lambda *_: rocm)
    monkeypatch.setattr(hardware, "_detect_intel", lambda *_: None)

    assert hardware.detect_machine().accelerator == "cuda"


def test_xpu_smi_memory_parser(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(hardware.shutil, "which", lambda command: "/usr/bin/xpu-smi" if command == "xpu-smi" else None)
    monkeypatch.setattr(
        hardware,
        "_run",
        lambda *_args, **_kwargs: "0, Intel Arc B580, 12288, 10240, 1.2.3\n1, Intel Max, 49152, 32768, 4.5.6",
    )

    devices = hardware._xpu_smi_devices()

    assert devices[0]["total"] == 12
    assert devices[1]["free"] == 32
