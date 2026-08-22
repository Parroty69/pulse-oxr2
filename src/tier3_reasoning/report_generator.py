from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:  # Transformers renamed the generic multimodal auto class.
        from transformers import AutoModelForMultimodalLM as AutoModelForImageTextToText
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoModelForImageTextToText = None
    AutoProcessor = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


class ReportGenerator:
    """Swappable backend: CheXagent-2-3b (CausalLM) or MedGemma-4B-it (image-text-to-text pipeline)."""

    def __init__(
        self,
        backend: str = "chexagent",
        device: str = "cuda",
        use_mock: bool = False,
        runtime: str = "transformers",
        quantization: str = "none",
        compute_dtype: str = "bf16",
        model_id: str | None = None,
    ):
        self.backend = backend
        self.use_mock = use_mock
        self.device = device
        self.runtime = runtime
        self.quantization = quantization
        self.compute_dtype = compute_dtype
        self.model_id = model_id
        self.tokenizer = None
        self.processor = None
        self.model = None
        self._mlx_generate = None
        self._mlx_apply_chat_template = None
        self._mlx_config = None

        if use_mock or backend == "mock":
            self.backend = "mock"
            self.runtime = "mock"
            return

        if runtime == "mlx-vlm":
            self._load_mlx()
        elif runtime == "transformers":
            self._load_transformers()
        else:
            raise ValueError(f"Unsupported Tier-3 runtime: {runtime}")

    def _dtype(self):
        if torch is None:
            return None
        normalized = self.compute_dtype.lower().replace("torch.", "")
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported Tier-3 compute dtype: {self.compute_dtype}")
        return mapping[normalized]

    def _quantization_config(self):
        if self.quantization == "none":
            return None
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes quantization was selected but Transformers support is unavailable")
        if self.quantization == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        if self.quantization == "nf4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self._dtype(),
            )
        raise ValueError(f"Unsupported Transformers quantization: {self.quantization}")

    def _load_transformers(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for the Transformers Tier-3 runtime")
        quantization_config = self._quantization_config()
        common_kwargs: dict[str, Any] = {
            "torch_dtype": self._dtype(),
            "low_cpu_mem_usage": True,
            "device_map": {"": self.device},
        }
        if quantization_config is not None:
            common_kwargs["quantization_config"] = quantization_config

        if self.backend == "chexagent":
            if AutoTokenizer is None or AutoModelForCausalLM is None:
                raise RuntimeError("Transformers does not expose the CheXagent loading classes")
            self.model_id = self.model_id or "StanfordAIMI/CheXagent-2-3b"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                **common_kwargs,
            ).eval()
        elif self.backend == "medgemma":
            if AutoProcessor is None or AutoModelForImageTextToText is None:
                raise RuntimeError("Transformers 4.50 or newer is required for MedGemma")
            self.model_id = self.model_id or "google/medgemma-4b-it"
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **common_kwargs).eval()
        else:
            raise ValueError(f"Unsupported Tier-3 backend: {self.backend}")

    def _load_mlx(self) -> None:
        if self.backend != "medgemma":
            raise ValueError("MLX-VLM is only configured for MedGemma in this repository")
        try:
            from mlx_vlm import generate as mlx_generate
            from mlx_vlm import load as mlx_load
            from mlx_vlm.prompt_utils import apply_chat_template
            from mlx_vlm.utils import load_config
        except ImportError as exc:  # pragma: no cover - only available on Apple Silicon deployment
            raise RuntimeError("mlx-vlm is required for the selected Apple Silicon runtime") from exc

        if not self.model_id:
            suffix = self.quantization if self.quantization in {"4bit", "6bit", "8bit"} else "bf16"
            self.model_id = f"mlx-community/medgemma-4b-it-{suffix}"
        self.model, self.processor = mlx_load(self.model_id)
        self._mlx_config = load_config(self.model_id)
        self._mlx_generate = mlx_generate
        self._mlx_apply_chat_template = apply_chat_template

    @staticmethod
    def _to_pil(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if np.issubdtype(array.dtype, np.floating):
            finite = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
            if finite.size and finite.max() <= 1.0:
                finite = finite * 255.0
            array = np.clip(finite, 0, 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    def _model_device(self):
        if self.model is None or torch is None:
            return self.device
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return self.device

    def _build_context(self, tier1_scores: list[dict], tier2_masks: list[dict]) -> str:
        top_scores = sorted(tier1_scores, key=lambda s: s.get("score", 0.0), reverse=True)[:8]
        score_lines = [f"- {s.get('label')}: {s.get('score', 0.0):.3f} @ {s.get('full_image_bbox')}" for s in top_scores]
        mask_lines = [
            f"- {m.get('label') or 'region'} score={m.get('score', 0.0):.3f} box={m.get('box')} area_px={m.get('area_px', 0)}"
            for m in tier2_masks
        ]
        return "\n".join(
            [
                "Tier 1 candidate findings:",
                *(score_lines or ["- none"]),
                "Tier 2 grounded masks:",
                *(mask_lines or ["- none"]),
            ]
        )

    def _mock_generate(self, tier1_scores: list[dict], tier2_masks: list[dict]) -> dict:
        findings = []
        if tier2_masks:
            for m in sorted(tier2_masks, key=lambda x: x.get("score", 0.0), reverse=True)[:3]:
                findings.append(
                    f"Possible {m.get('label', 'abnormality')} with spatial grounding at box {m.get('box')} (area {m.get('area_px', 0)} px)."
                )
        else:
            findings.append("No spatially grounded focal abnormality was detected by the current model thresholds.")

        impression = (
            "AI-assisted draft for clinician review: correlate with full clinical context and physician interpretation."
        )
        return {
            "findings": " ".join(findings),
            "impression": impression,
            "draft_status": "AI-Assisted Draft — Requires Physician Review",
            "grounding_context": self._build_context(tier1_scores, tier2_masks),
        }

    def generate_report(self, global_thumbnail, tier1_scores: list[dict], tier2_masks: list[dict]) -> dict:
        if self.backend == "mock":
            return self._mock_generate(tier1_scores, tier2_masks)

        grounding = self._build_context(tier1_scores, tier2_masks)
        prompt = (
            "You are an expert radiology assistant drafting a report for physician review only; "
            "do not produce a final diagnosis. Output sections FINDINGS and IMPRESSION.\n\n"
            f"Grounding context:\n{grounding}\n"
        )

        image = self._to_pil(global_thumbnail)
        if self.runtime == "mlx-vlm":
            formatted_prompt = self._mlx_apply_chat_template(
                self.processor,
                self._mlx_config,
                prompt,
                num_images=1,
            )
            result = self._mlx_generate(
                self.model,
                self.processor,
                formatted_prompt,
                [image],
                max_tokens=180,
                temperature=0.0,
                verbose=False,
            )
            text = result.text if hasattr(result, "text") else str(result)
        elif self.backend == "chexagent" and self.model is not None and self.tokenizer is not None:
            with TemporaryDirectory(prefix="cxr-chexagent-") as directory:
                image_path = Path(directory) / "thumbnail.png"
                image.save(image_path)
                query = self.tokenizer.from_list_format(
                    [{"image": str(image_path)}, {"text": prompt}]
                )
                conversation = [
                    {"from": "system", "value": "You are a careful radiology drafting assistant."},
                    {"from": "human", "value": query},
                ]
                input_ids = self.tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self._model_device())
                output = self.model.generate(
                    input_ids,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    max_new_tokens=180,
                )[0]
                text = self.tokenizer.decode(output[input_ids.size(1) :], skip_special_tokens=True)
        elif self.backend == "medgemma" and self.model is not None and self.processor is not None:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a careful radiology drafting assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image},
                    ],
                },
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model_device())
            output = self.model.generate(**inputs, max_new_tokens=180, do_sample=False)
            generated = output[0][inputs["input_ids"].shape[-1] :]
            text = self.processor.decode(generated, skip_special_tokens=True)
        else:
            raise RuntimeError("Tier-3 model was not initialized")

        findings, impression = _parse_sections(text)
        return {
            "findings": findings,
            "impression": impression,
            "draft_status": "AI-Assisted Draft — Requires Physician Review",
            "grounding_context": grounding,
        }


def _parse_sections(text: str) -> tuple[str, str]:
    upper = text.upper()
    f_idx = upper.find("FINDINGS")
    i_idx = upper.find("IMPRESSION")
    if f_idx == -1 and i_idx == -1:
        return text.strip(), ""
    if i_idx == -1:
        findings = text[f_idx:].split(":", 1)[-1].strip()
        return findings, ""
    findings_block = text[f_idx:i_idx] if f_idx != -1 else ""
    impression_block = text[i_idx:]
    findings = findings_block.split(":", 1)[-1].strip()
    impression = impression_block.split(":", 1)[-1].strip()
    return findings, impression
