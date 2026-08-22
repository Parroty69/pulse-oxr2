# Hardware acceleration and quantization

This document records the deployment policy for the models currently used by CXR Co-Pilot. Support was rechecked against primary project documentation on 2026-08-22.

## What is accelerated

| Pipeline tier | Model | Native implementation in this repo | Deployment policy |
|---|---|---|---|
| Tier 1 | [Microsoft BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | OpenCLIP/PyTorch | FP16 on GPU, FP32 on CPU. Not weight-quantized. |
| Tier 2 | [MedSAM ViT-B](https://github.com/bowang-lab/MedSAM) | Segment Anything/PyTorch | FP32 on the selected PyTorch device. Not weight-quantized. |
| Tier 3 | [CheXagent-2-3b](https://huggingface.co/StanfordAIMI/CheXagent-2-3b) | Transformers custom CausalLM | BF16/FP16, bitsandbytes INT8, or bitsandbytes NF4. |
| Tier 3 | [MedGemma-4b-it](https://huggingface.co/google/medgemma-4b-it) | Transformers Gemma 3 VLM, or MLX-VLM on Apple | BF16/FP16, bitsandbytes INT8/NF4, or an MLX 4/6/8-bit conversion. |

BiomedCLIP's published checkpoint is about 784 MB in FP32. MedSAM ViT-B has about 94 million parameters and its official [Zenodo checkpoint](https://zenodo.org/records/10689643) is 375 MB. Quantizing these two models saves relatively little compared with Tier 3 and would add an unvalidated accuracy change to screening and mask generation. The deployer therefore quantizes only the generative model.

The orchestration layer releases each model and clears its accelerator cache before loading the next tier. Quantization decisions are consequently based on the largest single tier plus runtime/activation headroom, rather than the sum of every model's weights.

## Platform mapping

| Machine | Selected stack | Notes |
|---|---|---|
| NVIDIA GPU | PyTorch CUDA + Transformers + bitsandbytes | Uses the official PyTorch CUDA wheel channel compatible with the driver-reported CUDA level. |
| AMD GPU | PyTorch ROCm/HIP + Transformers + bitsandbytes | ROCm deliberately uses `cuda` and `torch.cuda` device names in PyTorch. ROCm must already support the exact GPU and OS. |
| Apple Silicon | PyTorch MPS for Tiers 1–2; MLX-VLM for MedGemma | Uses Metal and unified memory. `PYTORCH_ENABLE_MPS_FALLBACK=1` allows unsupported MPS operators to fall back to CPU. |
| Intel GPU | Native PyTorch XPU/SYCL + Transformers + bitsandbytes | **XPU is the current Intel analogue to CUDA.** Stock PyTorch has native Intel GPU support; the wheel comes from the official `xpu` index. |
| CPU only | PyTorch CPU + Transformers + bitsandbytes | Functional fallback, but Tier-3 generation is expected to be slow. |

Primary runtime references:

- [PyTorch local installation selector](https://pytorch.org/get-started/locally/)
- [PyTorch HIP/ROCm semantics](https://docs.pytorch.org/docs/stable/notes/hip.html)
- [PyTorch XPU API](https://docs.pytorch.org/docs/stable/xpu.html)
- [Intel's native PyTorch XPU overview](https://www.intel.com/content/www/us/en/developer/tools/oneapi/optimization-for-pytorch.html)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm)
- [bitsandbytes installation and hardware support](https://huggingface.co/docs/bitsandbytes/installation)
- [AMD ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)

OpenVINO is Intel's optimized inference/deployment alternative and can target Intel CPU, GPU, and NPU. It is not the automatic choice here because this repository combines OpenCLIP, custom SAM code, and two multimodal Transformers paths. Native PyTorch XPU keeps those components on one execution API and avoids maintaining three separately converted OpenVINO graphs. OpenVINO remains a sensible future serving target after model-specific export and output-equivalence validation.

## Automatic memory policy

The detector uses free VRAM for NVIDIA, driver/sysfs VRAM data for AMD, XPU Manager data for Intel GPUs, and an OS-reserved unified-memory budget for Apple or integrated GPUs. On multi-vendor hosts it selects a runtime-ready accelerator with the most free memory (CUDA wins an equal-memory tie); `--accelerator` provides an explicit override.

Transformers policy:

| Available accelerator memory | MedGemma | CheXagent |
|---|---|---|
| At least 13 GiB | BF16/FP16 | BF16/FP16 |
| 10–13 GiB | INT8 | BF16/FP16 |
| 8–10 GiB | INT8 | INT8 |
| Less than 8 GiB | NF4 4-bit | NF4 4-bit |

CPU uses INT8 when at least 24 GiB of system memory is currently available and NF4 otherwise. Low-bit CPU execution reduces memory, not latency.

MLX-VLM policy on Apple unified memory:

| Available unified-memory budget | Selected MedGemma artifact |
|---|---|
| At least 20 GiB | BF16 |
| 14–20 GiB | 8-bit |
| 10–14 GiB | 6-bit |
| Less than 10 GiB | 4-bit |

The MLX artifacts are the `mlx-community/medgemma-4b-it-{4bit,6bit,8bit,bf16}` conversions. They derive from Google's licensed weights but are community conversions, so deployment owners should validate output parity for their intended research workflow.

The thresholds are deliberately conservative and leave room for image tensors, KV cache, Streamlit, and the operating system. They are defaults, not claims of clinical equivalence between precisions. Every selected precision still requires task-specific validation.

For NVIDIA, the selector also enforces bitsandbytes hardware floors: LLM.int8() requires compute capability 7.5, while NF4 requires 6.0. A Pascal-class device that lands in the INT8 memory band is moved to NF4 automatically; older devices receive an explicit CPU-fallback instruction.

## One-click usage

Normal deployment:

```bash
./deploy.sh
```

Useful controls:

```bash
# Inspect detection and policy only
./deploy.sh --detect-only

# Show installs, model choice, and generated config without making changes
./deploy.sh --dry-run --no-launch

# Prepare everything but do not start the UI
./deploy.sh --no-launch

# Skip prefetching (requires an existing MedSAM checkpoint; Hugging Face loads lazily)
./deploy.sh --skip-model-download

# Explicit model/precision override
./deploy.sh --backend chexagent --quantization nf4

# Force a CPU deployment even when a GPU is present
./deploy.sh --accelerator cpu
```

The generated `config/runtime.generated.yaml` is a JSON-formatted YAML overlay and is intentionally ignored by Git. The UI recursively merges it over `config/pipeline_config.yaml`. Set `CXR_PIPELINE_CONFIG=/absolute/path/to/another.yaml` to use a different overlay.

For reproducible infrastructure tests, `--profile-json` accepts either a JSON file or a literal serialized `MachineProfile`. It is an offline simulation feature; it does not bypass the post-install accelerator probe.

## Prerequisites and intentional boundaries

- Python 3.10 or newer is required.
- NVIDIA drivers, ROCm, and Intel Level Zero compute drivers are system components. The script does not install or modify kernel drivers.
- AMD auto-detection currently targets native Linux ROCm. AMD's Windows/WSL support has a different hardware-specific installation matrix; use a validated ROCm image or complete that system setup before invoking the Python deployer.
- MedGemma files are gated. Accept the Health AI Developer Foundations terms with the deployment Hugging Face account and export `HF_TOKEN` before prefetching.
- Jetson uses JetPack/L4T ABI-specific packages and is not treated as a generic Linux CUDA host by this installer. Use NVIDIA's Jetson PyTorch packages and run with `--skip-install` after creating a compatible environment.
- If runtime verification fails, deployment stops instead of silently switching a requested real model to the mock backend.
