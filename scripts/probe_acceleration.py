#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the installed accelerator runtime")
    parser.add_argument("--hardware-backend", required=True, choices=["cuda", "rocm", "mps", "xpu", "cpu"])
    parser.add_argument("--framework", required=True, choices=["transformers", "mlx-vlm", "mock"])
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    import torch

    result: dict[str, object] = {
        "torch": torch.__version__,
        "hardware_backend": args.hardware_backend,
        "framework": args.framework,
    }
    runtime_device = torch.device("cpu")
    if args.hardware_backend in {"cuda", "rocm"}:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        is_rocm = bool(getattr(torch.version, "hip", None))
        if args.hardware_backend == "rocm" and not is_rocm:
            raise RuntimeError("A CUDA build of PyTorch was installed instead of a ROCm/HIP build")
        if args.hardware_backend == "cuda" and is_rocm:
            raise RuntimeError("A ROCm/HIP build of PyTorch was installed instead of a CUDA build")
        properties = torch.cuda.get_device_properties(args.device_index)
        result.update(
            {
                "device": properties.name,
                "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                "cuda": getattr(torch.version, "cuda", None),
                "hip": getattr(torch.version, "hip", None),
            }
        )
        runtime_device = torch.device(f"cuda:{args.device_index}")
        probe = torch.ones((2, 2), device=runtime_device) @ torch.ones(
            (2, 2), device=runtime_device
        )
        torch.cuda.synchronize(args.device_index)
        result["probe_sum"] = float(probe.sum().cpu())
    elif args.hardware_backend == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("torch.backends.mps.is_available() is false")
        result["device"] = "Apple Metal Performance Shaders"
        runtime_device = torch.device("mps")
        probe = torch.ones((2, 2), device=runtime_device) @ torch.ones(
            (2, 2), device=runtime_device
        )
        torch.mps.synchronize()
        result["probe_sum"] = float(probe.sum().cpu())
    elif args.hardware_backend == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("torch.xpu.is_available() is false")
        properties = torch.xpu.get_device_properties(args.device_index)
        result.update(
            {
                "device": properties.name,
                "total_memory_gib": round(properties.total_memory / 1024**3, 2),
            }
        )
        runtime_device = torch.device(f"xpu:{args.device_index}")
        probe = torch.ones((2, 2), device=runtime_device) @ torch.ones(
            (2, 2), device=runtime_device
        )
        torch.xpu.synchronize(args.device_index)
        result["probe_sum"] = float(probe.sum().cpu())
    else:
        result["device"] = "CPU"

    if args.framework == "mlx-vlm":
        import mlx.core as mx
        import mlx_vlm

        result["mlx_device"] = str(mx.default_device())
        result["mlx_vlm"] = getattr(mlx_vlm, "__version__", "installed")
        result["mlx_probe_sum"] = float(mx.sum(mx.ones((2, 2))).item())
    if args.framework == "transformers" and args.quantization in {"int8", "nf4"}:
        import bitsandbytes

        result["bitsandbytes"] = getattr(bitsandbytes, "__version__", "installed")
        probe_dtype = torch.float32 if args.hardware_backend == "cpu" else torch.float16
        if args.quantization == "int8":
            quantized_layer = bitsandbytes.nn.Linear8bitLt(16, 8, bias=False)
        else:
            quantized_layer = bitsandbytes.nn.Linear4bit(
                16,
                8,
                bias=False,
                compute_dtype=probe_dtype,
                quant_type="nf4",
            )
        quantized_layer = quantized_layer.to(runtime_device)
        quantized_output = quantized_layer(
            torch.ones((1, 16), device=runtime_device, dtype=probe_dtype)
        )
        result["bitsandbytes_probe_shape"] = list(quantized_output.shape)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
