from __future__ import annotations

import csv
import ctypes
import json
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


GIB = 1024**3


@dataclass(frozen=True)
class MachineProfile:
    """Hardware facts available before a platform-specific PyTorch install."""

    os_name: str
    architecture: str
    accelerator: str
    device_name: str
    torch_device: str
    total_memory_gib: float
    available_memory_gib: float
    system_memory_gib: float
    system_available_memory_gib: float
    device_index: int = 0
    unified_memory: bool = False
    driver_version: str | None = None
    runtime_version: str | None = None
    compute_capability: str | None = None
    runtime_ready: bool = True
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MachineProfile":
        payload = dict(value)
        payload["warnings"] = tuple(payload.get("warnings", ()))
        return cls(**payload)


@dataclass(frozen=True)
class DeploymentPlan:
    backend: str
    framework: str
    quantization: str
    model_id: str
    hardware_backend: str
    torch_device: str
    torch_dtype: str
    screening_dtype: str
    tile_batch_size: int
    release_between_tiers: bool
    use_mock_models: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(command: Iterable[str], timeout: int = 8) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _read_float(path: Path) -> float | None:
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _parse_version(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+){1,3})", text)
    return match.group(1) if match else None


@lru_cache(maxsize=1)
def _apple_hardware_overview() -> dict[str, Any]:
    output = _run(["system_profiler", "SPHardwareDataType", "-json"], timeout=15)
    if not output:
        return {}
    try:
        entries = json.loads(output).get("SPHardwareDataType", [])
    except (json.JSONDecodeError, AttributeError):
        return {}
    return entries[0] if entries else {}


def _memory_string_to_gib(value: str) -> float:
    match = re.search(r"([0-9.]+)\s*(GB|TB|MB)", value, re.IGNORECASE)
    if not match:
        return 0.0
    amount = float(match.group(1))
    unit = match.group(2).upper()
    return amount * 1024 if unit == "TB" else amount / 1024 if unit == "MB" else amount


def _system_memory() -> tuple[float, float]:
    os_name = platform.system().lower()
    if os_name == "linux":
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return 0.0, 0.0
        return values.get("MemTotal", 0) / GIB, values.get("MemAvailable", 0) / GIB

    if os_name == "darwin":
        total_raw = _run(["sysctl", "-n", "hw.memsize"])
        total = float(total_raw) / GIB if total_raw and total_raw.isdigit() else 0.0
        if not total:
            total = _memory_string_to_gib(str(_apple_hardware_overview().get("physical_memory", "")))
        vm_stat = _run(["vm_stat"])
        if not vm_stat:
            return total, max(0.0, total * 0.75)
        page_match = re.search(r"page size of (\d+) bytes", vm_stat)
        page_size = int(page_match.group(1)) if page_match else 4096
        pages = 0
        for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
            match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.", vm_stat, re.MULTILINE)
            if match:
                pages += int(match.group(1))
        available = pages * page_size / GIB
        return total, min(total, available) if available else max(0.0, total * 0.75)

    if os_name == "windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        except (AttributeError, OSError):
            return 0.0, 0.0
        return status.total_physical / GIB, status.available_physical / GIB

    return 0.0, 0.0


def _detect_nvidia(system_total: float, system_available: float) -> MachineProfile | None:
    if shutil.which("nvidia-smi") is None:
        return None
    fields = "index,name,memory.total,memory.free,driver_version,compute_cap"
    output = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    with_compute_capability = True
    if not output:
        fields = "index,name,memory.total,memory.free,driver_version"
        output = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        with_compute_capability = False
    if not output:
        return None

    rows = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        try:
            rows.append(
                {
                    "index": int(row[0]),
                    "name": row[1].strip(),
                    "total": float(row[2]) / 1024,
                    "free": float(row[3]) / 1024,
                    "driver": row[4].strip(),
                    "compute": row[5].strip() if with_compute_capability and len(row) > 5 else None,
                }
            )
        except (ValueError, IndexError):
            continue
    if not rows:
        return None
    selected = max(rows, key=lambda item: item["free"])
    banner = _run(["nvidia-smi"])
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", banner or "")
    is_jetson = platform.machine().lower() in {"arm64", "aarch64"} and Path(
        "/etc/nv_tegra_release"
    ).exists()
    warnings: tuple[str, ...] = ()
    if is_jetson:
        warnings = (
            "Jetson/JetPack uses L4T-specific PyTorch and bitsandbytes builds; prepare that environment and rerun with --skip-install.",
        )
    return MachineProfile(
        os_name=platform.system().lower(),
        architecture=platform.machine().lower(),
        accelerator="cuda",
        device_name=selected["name"],
        torch_device=f"cuda:{selected['index']}",
        total_memory_gib=selected["total"],
        available_memory_gib=selected["free"],
        system_memory_gib=system_total,
        system_available_memory_gib=system_available,
        device_index=selected["index"],
        driver_version=selected["driver"],
        runtime_version=cuda_match.group(1) if cuda_match else None,
        compute_capability=selected["compute"],
        runtime_ready=not is_jetson,
        warnings=warnings,
    )


def _gpu_cards(vendor_id: str) -> list[Path]:
    cards: list[Path] = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        try:
            vendor = (card / "device/vendor").read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if vendor == vendor_id:
            cards.append(card)
    return cards


def _pci_device_name(vendor_pattern: str, fallback: str) -> str:
    output = _run(["lspci", "-nn"])
    if output:
        for line in output.splitlines():
            if re.search(vendor_pattern, line, re.IGNORECASE) and re.search(
                r"(VGA|Display|3D controller)", line, re.IGNORECASE
            ):
                return line.split(": ", 1)[-1].strip()
    return fallback


def _card_memory(card: Path, system_total: float, system_available: float) -> tuple[float, float, bool]:
    total = _read_float(card / "device/mem_info_vram_total")
    used = _read_float(card / "device/mem_info_vram_used")
    if total and total > 0:
        free = max(0.0, total - (used or 0.0))
        return total / GIB, free / GIB, False
    return system_total, system_available, True


def _detect_amd(system_total: float, system_available: float) -> MachineProfile | None:
    cards = _gpu_cards("0x1002")
    rocm_tool = shutil.which("rocminfo") or shutil.which("rocm-smi") or shutil.which("amd-smi")
    if not cards and not rocm_tool:
        return None

    memories = [_card_memory(card, system_total, system_available) for card in cards]
    if memories:
        index, memory = max(enumerate(memories), key=lambda item: item[1][1])
        total, available, unified = memory
        if total < 2 and system_total >= 8:
            # ROCm-capable Ryzen APUs expose shared memory rather than useful
            # dedicated VRAM through this sysfs counter.
            total, available, unified = system_total, system_available, True
    else:
        index, total, available, unified = 0, 0.0, 0.0, False

    version_text = None
    for candidate in (
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/.info/version-dev"),
    ):
        try:
            version_text = candidate.read_text(encoding="utf-8")
            break
        except OSError:
            continue
    if not version_text and shutil.which("hipconfig"):
        version_text = _run(["hipconfig", "--version"])
    ready = Path("/dev/kfd").exists() and bool(rocm_tool or Path("/opt/rocm").exists())
    warnings: tuple[str, ...] = ()
    if not ready:
        warnings = (
            "An AMD GPU was found, but the ROCm userspace/driver was not detected. Install a supported ROCm stack first.",
        )
    return MachineProfile(
        os_name=platform.system().lower(),
        architecture=platform.machine().lower(),
        accelerator="rocm",
        device_name=_pci_device_name(r"AMD|ATI", "AMD GPU"),
        # PyTorch deliberately reuses CUDA device strings for HIP/ROCm.
        torch_device=f"cuda:{index}",
        total_memory_gib=total,
        available_memory_gib=available,
        system_memory_gib=system_total,
        system_available_memory_gib=system_available,
        device_index=index,
        unified_memory=unified,
        runtime_version=_parse_version(version_text),
        runtime_ready=ready,
        warnings=warnings,
    )


def _xpu_smi_devices() -> list[dict[str, Any]]:
    if shutil.which("xpu-smi") is None:
        return []
    output = _run(
        [
            "xpu-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return []
    devices: list[dict[str, Any]] = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        try:
            devices.append(
                {
                    "index": int(row[0]),
                    "name": row[1].strip(),
                    # xpu-smi reports memory.total/free in MiB.
                    "total": float(row[2]) / 1024,
                    "free": float(row[3]) / 1024,
                    "driver": row[4].strip(),
                }
            )
        except (IndexError, ValueError):
            continue
    return devices


def _detect_intel(system_total: float, system_available: float) -> MachineProfile | None:
    cards = _gpu_cards("0x8086")
    xpu_devices = _xpu_smi_devices()
    if not cards and not xpu_devices:
        return None
    if xpu_devices:
        selected = max(xpu_devices, key=lambda item: item["free"])
        index = selected["index"]
        total = selected["total"]
        available = selected["free"]
        device_name = selected["name"]
        driver_version = selected["driver"]
        unified = not any(_read_float(card / "device/mem_info_vram_total") for card in cards)
    else:
        memories = [_card_memory(card, system_total, system_available) for card in cards]
        index, memory = max(enumerate(memories), key=lambda item: item[1][1])
        total, available, unified = memory
        device_name = _pci_device_name(r"Intel", "Intel GPU")
        driver_version = None
    render_nodes = list(Path("/dev/dri").glob("renderD*"))
    ready = bool(render_nodes or xpu_devices)
    warnings: tuple[str, ...] = ()
    if not ready:
        warnings = (
            "An Intel GPU was found, but a Level Zero render device was not detected. Install the Intel GPU compute driver first.",
        )
    return MachineProfile(
        os_name=platform.system().lower(),
        architecture=platform.machine().lower(),
        accelerator="xpu",
        device_name=device_name,
        torch_device=f"xpu:{index}",
        total_memory_gib=total,
        available_memory_gib=available,
        system_memory_gib=system_total,
        system_available_memory_gib=system_available,
        device_index=index,
        unified_memory=unified,
        driver_version=driver_version,
        runtime_ready=ready,
        warnings=warnings,
    )


def _cpu_profile(system_total: float, system_available: float) -> MachineProfile:
    return MachineProfile(
        os_name=platform.system().lower(),
        architecture=platform.machine().lower(),
        accelerator="cpu",
        device_name=platform.processor() or platform.machine() or "CPU",
        torch_device="cpu",
        total_memory_gib=system_total,
        available_memory_gib=system_available,
        system_memory_gib=system_total,
        system_available_memory_gib=system_available,
        unified_memory=True,
    )


def detect_machine(preferred_accelerator: str | None = None) -> MachineProfile:
    """Detect the best available local inference accelerator without importing torch."""

    system_total, system_available = _system_memory()
    os_name = platform.system().lower()
    architecture = platform.machine().lower()
    preferred = (preferred_accelerator or "").lower()
    preferred = "mps" if preferred == "mlx" else preferred
    if preferred == "cpu":
        return _cpu_profile(system_total, system_available)

    if os_name == "darwin" and architecture in {"arm64", "aarch64"}:
        # Apple GPUs use unified memory. Reserve memory for macOS and the UI rather
        # than treating all physical RAM as an inference budget.
        budget = min(system_available, max(0.0, system_total - max(4.0, system_total * 0.25)))
        if budget <= 0 and system_total:
            budget = max(0.0, system_total * 0.65)
        apple = MachineProfile(
            os_name=os_name,
            architecture=architecture,
            accelerator="mps",
            device_name=str(_apple_hardware_overview().get("chip_type") or platform.processor() or "Apple Silicon GPU"),
            torch_device="mps",
            total_memory_gib=system_total,
            available_memory_gib=budget,
            system_memory_gib=system_total,
            system_available_memory_gib=system_available,
            unified_memory=True,
        )
        if preferred and preferred != "mps":
            raise ValueError(f"Requested {preferred!r}, but this Apple Silicon machine exposes MPS/MLX")
        return apple

    candidates: list[MachineProfile] = []
    nvidia = _detect_nvidia(system_total, system_available)
    if nvidia:
        candidates.append(nvidia)
    if os_name == "linux":
        amd = _detect_amd(system_total, system_available)
        if amd:
            candidates.append(amd)
    if os_name in {"linux", "windows"}:
        intel = _detect_intel(system_total, system_available)
        if intel:
            candidates.append(intel)

    if preferred:
        for candidate in candidates:
            if candidate.accelerator == preferred:
                return candidate
        raise ValueError(f"Requested accelerator {preferred!r} was not detected")
    if candidates:
        ready = [candidate for candidate in candidates if candidate.runtime_ready]
        pool = ready or candidates
        priority = {"cuda": 3, "rocm": 2, "xpu": 1}
        return max(
            pool,
            key=lambda candidate: (
                candidate.available_memory_gib,
                priority.get(candidate.accelerator, 0),
            ),
        )

    return _cpu_profile(system_total, system_available)


def force_accelerator(profile: MachineProfile, accelerator: str) -> MachineProfile:
    """Apply a safe user override after auto-detection."""

    accelerator = accelerator.lower()
    if accelerator == profile.accelerator:
        return profile
    if accelerator == "cpu":
        return replace(
            profile,
            accelerator="cpu",
            device_name=platform.processor() or "CPU",
            torch_device="cpu",
            total_memory_gib=profile.system_memory_gib,
            available_memory_gib=profile.system_available_memory_gib,
            device_index=0,
            unified_memory=True,
            driver_version=None,
            runtime_version=None,
            compute_capability=None,
            runtime_ready=True,
            warnings=(),
        )
    if accelerator in {"mlx", "mps"} and profile.os_name == "darwin" and profile.architecture in {
        "arm64",
        "aarch64",
    }:
        return replace(profile, accelerator="mps", torch_device="mps")
    raise ValueError(
        f"Cannot force {accelerator!r}: detected {profile.accelerator!r}. "
        "Use a JSON profile only for offline simulation."
    )


def _compute_dtype(profile: MachineProfile) -> str:
    if profile.accelerator == "cuda":
        try:
            major = int((profile.compute_capability or "0").split(".", 1)[0])
        except ValueError:
            major = 0
        return "bf16" if major >= 8 else "fp16"
    if profile.accelerator == "rocm":
        return "bf16" if re.search(r"MI(?:2|3)|gfx9", profile.device_name, re.IGNORECASE) else "fp16"
    if profile.accelerator in {"mps", "xpu"}:
        return "fp16"
    return "fp32"


def _cuda_compute_capability(profile: MachineProfile) -> tuple[int, int] | None:
    if not profile.compute_capability:
        return None
    try:
        parts = profile.compute_capability.split(".", 1)
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None


def _auto_transformers_quantization(profile: MachineProfile, backend: str) -> str:
    memory = profile.available_memory_gib or profile.total_memory_gib
    if profile.accelerator == "cpu":
        memory = profile.system_available_memory_gib or profile.system_memory_gib
        return "int8" if memory >= 24 else "nf4"

    full_precision_floor = 13.0 if backend == "medgemma" else 10.0
    if memory >= full_precision_floor:
        return "none"
    if memory >= 8.0:
        return "int8"
    return "nf4"


def _auto_mlx_quantization(profile: MachineProfile) -> str:
    memory = profile.available_memory_gib or profile.total_memory_gib
    if memory >= 20:
        return "bf16"
    if memory >= 14:
        return "8bit"
    if memory >= 10:
        return "6bit"
    return "4bit"


def _normalize_quantization(value: str, framework: str) -> str:
    value = value.lower().replace("-", "")
    if framework == "mlx-vlm":
        aliases = {
            "none": "bf16",
            "bf16": "bf16",
            "int8": "8bit",
            "8bit": "8bit",
            "6bit": "6bit",
            "4bit": "4bit",
            "nf4": "4bit",
        }
    else:
        aliases = {
            "none": "none",
            "bf16": "none",
            "fp16": "none",
            "int8": "int8",
            "8bit": "int8",
            "4bit": "nf4",
            "nf4": "nf4",
        }
    if value not in aliases:
        supported = ", ".join(sorted(aliases))
        raise ValueError(f"Quantization {value!r} is not valid for {framework}; choose one of: {supported}")
    return aliases[value]


def select_deployment_plan(
    profile: MachineProfile,
    backend: str = "auto",
    framework: str = "auto",
    quantization: str = "auto",
) -> DeploymentPlan:
    """Choose the Tier-3 runtime and quantization while leaving vision tiers in floating point."""

    backend = backend.lower()
    if backend == "auto":
        backend = "medgemma"
    if backend not in {"medgemma", "chexagent", "mock"}:
        raise ValueError(f"Unsupported Tier-3 backend: {backend}")

    if backend == "mock":
        return DeploymentPlan(
            backend="mock",
            framework="mock",
            quantization="none",
            model_id="",
            hardware_backend=profile.accelerator,
            torch_device=profile.torch_device,
            torch_dtype=_compute_dtype(profile),
            screening_dtype="fp32" if profile.accelerator == "cpu" else "fp16",
            tile_batch_size=1,
            release_between_tiers=True,
            use_mock_models=True,
            warnings=profile.warnings,
        )

    framework = framework.lower()
    if framework == "auto":
        framework = "mlx-vlm" if profile.accelerator == "mps" and backend == "medgemma" else "transformers"
    aliases = {"mlx": "mlx-vlm", "pytorch": "transformers", "torch": "transformers"}
    framework = aliases.get(framework, framework)
    if framework not in {"mlx-vlm", "transformers"}:
        raise ValueError(f"Unsupported framework: {framework}")
    if framework == "mlx-vlm" and profile.accelerator != "mps":
        raise ValueError("MLX-VLM requires an Apple Silicon machine")
    if framework == "mlx-vlm" and backend != "medgemma":
        raise ValueError("CheXagent's custom multimodal architecture is not supported by MLX-VLM")

    policy_warnings: list[str] = []
    if quantization == "auto":
        if framework == "mlx-vlm":
            selected_quantization = _auto_mlx_quantization(profile)
        elif profile.accelerator == "mps":
            selected_quantization = "none"
            memory = profile.available_memory_gib or profile.total_memory_gib
            full_precision_floor = 13.0 if backend == "medgemma" else 10.0
            if memory < full_precision_floor:
                policy_warnings.append(
                    "Low-bit Transformers loading is not enabled on Apple MPS; the model will use FP16. "
                    "Free unified memory or select MedGemma with MLX-VLM if it does not fit."
                )
        else:
            selected_quantization = _auto_transformers_quantization(profile, backend)
    else:
        selected_quantization = _normalize_quantization(quantization, framework)
        if (
            framework == "transformers"
            and profile.accelerator == "mps"
            and selected_quantization != "none"
        ):
            raise ValueError(
                "Low-bit Transformers quantization is not enabled on Apple MPS; "
                "use --quantization none or select MedGemma with MLX-VLM"
            )

    capability = _cuda_compute_capability(profile)
    if framework == "transformers" and profile.accelerator == "cuda" and capability:
        if selected_quantization == "int8" and capability < (7, 5):
            if quantization == "auto" and capability >= (6, 0):
                selected_quantization = "nf4"
                policy_warnings.append(
                    "This NVIDIA GPU is below compute capability 7.5, so NF4 was selected instead of LLM.int8()."
                )
            else:
                raise ValueError("bitsandbytes LLM.int8() requires NVIDIA compute capability 7.5 or newer")
        if selected_quantization == "nf4" and capability < (6, 0):
            raise ValueError(
                "bitsandbytes NF4 requires NVIDIA compute capability 6.0 or newer; rerun with --accelerator cpu"
            )

    if framework == "mlx-vlm":
        model_id = {
            "bf16": "mlx-community/medgemma-4b-it-bf16",
            "8bit": "mlx-community/medgemma-4b-it-8bit",
            "6bit": "mlx-community/medgemma-4b-it-6bit",
            "4bit": "mlx-community/medgemma-4b-it-4bit",
        }[selected_quantization]
    else:
        model_id = {
            "medgemma": "google/medgemma-4b-it",
            "chexagent": "StanfordAIMI/CheXagent-2-3b",
        }[backend]

    memory = profile.available_memory_gib or profile.total_memory_gib
    if profile.accelerator == "cpu":
        memory = profile.system_available_memory_gib or profile.system_memory_gib
    tile_batch_size = 4 if memory >= 12 else 2 if memory >= 8 else 1
    warnings = [*profile.warnings, *policy_warnings]
    if memory and memory < 5:
        warnings.append(
            "Less than 5 GiB is currently available; 4-bit loading may still fail and CPU inference will be slow."
        )
    if profile.accelerator == "cpu":
        warnings.append("No supported GPU was selected; Tier-3 inference will be substantially slower on CPU.")
    if framework == "mlx-vlm":
        warnings.append(
            "The MLX MedGemma artifacts are community conversions of Google's licensed weights; validate them before clinical research use."
        )

    dtype = _compute_dtype(profile)
    return DeploymentPlan(
        backend=backend,
        framework=framework,
        quantization=selected_quantization,
        model_id=model_id,
        hardware_backend=profile.accelerator,
        torch_device=profile.torch_device,
        torch_dtype=dtype,
        screening_dtype="fp32" if profile.accelerator == "cpu" else "fp16",
        tile_batch_size=tile_batch_size,
        release_between_tiers=True,
        warnings=tuple(warnings),
    )


def pytorch_index_url(profile: MachineProfile) -> str | None:
    """Return the official PyTorch wheel index appropriate to the detected stack."""

    if profile.accelerator == "cuda":
        version = tuple(int(part) for part in (_parse_version(profile.runtime_version) or "0.0").split(".")[:2])
        if version >= (13, 2):
            channel = "cu132"
        elif version >= (13, 0):
            channel = "cu130"
        else:
            # CUDA 12.6 is the broadest current stable wheel baseline. The
            # post-install probe provides a clear failure for an older driver.
            channel = "cu126"
        return f"https://download.pytorch.org/whl/{channel}"
    if profile.accelerator == "rocm":
        version = tuple(int(part) for part in (_parse_version(profile.runtime_version) or "7.2").split(".")[:2])
        if version >= (7, 2):
            channel = "rocm7.2"
        elif version >= (7, 1):
            channel = "rocm7.1"
        else:
            channel = "rocm6.4"
        return f"https://download.pytorch.org/whl/{channel}"
    if profile.accelerator == "xpu":
        return "https://download.pytorch.org/whl/xpu"
    if profile.accelerator == "cpu" and profile.os_name == "linux":
        return "https://download.pytorch.org/whl/cpu"
    return None


def profile_from_json(path_or_json: str) -> MachineProfile:
    candidate = Path(path_or_json)
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    if is_file:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path_or_json)
    return MachineProfile.from_dict(payload)
