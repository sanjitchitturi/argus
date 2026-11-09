"""
Argus VLM Module — BLIP-2 / LLaVA Image Captioning & VQA
"""

from __future__ import annotations
import time
import os
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from loguru import logger

try:
    import os

    if os.getenv("ARGUS_MODE", "demo") == "demo":
        raise ImportError("Explicit demo mode")
    import torch
    from transformers import (
        Blip2Processor,
        Blip2ForConditionalGeneration,
        AutoProcessor,
        LlavaForConditionalGeneration,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — VLM running in mock mode")


# ── Standard VQA questions for industrial inspection ────────────────────────
DEFAULT_QUESTIONS = [
    "What defect is visible in this image?",
    "Where is the defect located in the image?",
    "How severe does the defect appear?",
    "What is the surface condition around the defect?",
    "Is there any discoloration or texture anomaly?",
]


class VisionLanguageModel:
    """
    Unified wrapper for BLIP-2 and LLaVA.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
        Use 'blip2' or 'llava' as shortcuts.
    device : str
        'cpu', 'cuda', 'mps'
    load_in_8bit : bool
        Enable 8-bit quantisation (requires bitsandbytes + CUDA).
    """

    SHORTCUTS = {
        "blip2": "Salesforce/blip2-opt-2.7b",
        "llava": "llava-hf/llava-1.5-7b-hf",
    }

    def __init__(
        self,
        model_name: str = "blip2",
        device: str = "cpu",
        load_in_8bit: bool = False,
    ):
        self.model_id = self.SHORTCUTS.get(model_name, model_name)
        self.model_type = "blip2" if "blip2" in self.model_id.lower() else "llava"
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.processor = None
        self.model = None
        self._load()

    # ── Loading ────────────────────────────────────────────────────────────

    def _load(self):
        if os.getenv("ARGUS_MODE", "demo") == "demo":
            return
        if os.getenv("ARGUS_MODE") != "model":
            raise ValueError("ARGUS_MODE must be demo or model")
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Model dependencies missing; install requirements-models.txt")
            logger.warning("Transformers unavailable — VLM in mock mode")
            return
        try:
            logger.info(f"Loading VLM: {self.model_id} on {self.device}")
            kwargs = {"torch_dtype": torch.float16 if self.device != "cpu" else torch.float32}
            if self.load_in_8bit:
                kwargs["load_in_8bit"] = True

            if self.model_type == "blip2":
                self.processor = Blip2Processor.from_pretrained(self.model_id)
                self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
            else:  # llava
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id, **kwargs)

            if not self.load_in_8bit:
                self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("VLM loaded successfully")
        except Exception as exc:
            logger.error(f"VLM load failed: {exc}")
            raise RuntimeError("Model loading failed; check dependencies and checkpoint") from exc

    # ── Caption ────────────────────────────────────────────────────────────

    def caption(self, image: Union[str, Path, np.ndarray, Image.Image]) -> dict:
        """Generate an image caption."""
        pil_img = self._to_pil(image)
        t0 = time.perf_counter()

        if self.model is None:
            text = self._mock_caption()
        elif self.model_type == "blip2":
            text = self._blip2_caption(pil_img)
        else:
            text = self._llava_caption(pil_img)

        return {
            "caption": text,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 2),
            "model": self.model_id if self.model is not None else "synthetic-demo",
        }

    def _blip2_caption(self, img: Image.Image) -> str:
        import torch

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(ids[0], skip_special_tokens=True).strip()

    def _llava_caption(self, img: Image.Image) -> str:
        import torch

        prompt = "USER: <image>\nDescribe this industrial inspection image in detail.\nASSISTANT:"
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=150)
        full = self.processor.decode(ids[0], skip_special_tokens=True)
        return full.split("ASSISTANT:")[-1].strip()

    @staticmethod
    def _mock_caption() -> str:
        return (
            "The image shows a metal surface with visible surface defects including "
            "linear cracks and localized corrosion patches near the center of the component. "
            "The surrounding material appears structurally intact."
        )

    # ── VQA ───────────────────────────────────────────────────────────────

    def answer(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        question: str,
    ) -> dict:
        """Answer a visual question about the image."""
        pil_img = self._to_pil(image)
        t0 = time.perf_counter()

        if self.model is None:
            answer = self._mock_answer(question)
        elif self.model_type == "blip2":
            answer = self._blip2_vqa(pil_img, question)
        else:
            answer = self._llava_vqa(pil_img, question)

        return {
            "question": question,
            "answer": answer,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def _blip2_vqa(self, img: Image.Image, question: str) -> str:
        import torch

        inputs = self.processor(images=img, text=question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=80)
        return self.processor.decode(ids[0], skip_special_tokens=True).strip()

    def _llava_vqa(self, img: Image.Image, question: str) -> str:
        import torch

        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=120)
        full = self.processor.decode(ids[0], skip_special_tokens=True)
        return full.split("ASSISTANT:")[-1].strip()

    @staticmethod
    def _mock_answer(question: str) -> str:
        q = question.lower()
        if "what" in q and "defect" in q:
            return "A longitudinal crack is visible, approximately 3–4 cm in length, with secondary micro-fractures radiating from the main fault line."
        if "where" in q or "location" in q or "located" in q:
            return "The defect is located in the upper-center region of the inspected surface, approximately 15% from the top edge."
        if "severe" in q or "severity" in q:
            return "The defect appears moderately severe. The crack depth cannot be confirmed visually, but surface propagation suggests it may require immediate intervention."
        if "color" in q or "discolor" in q:
            return "There is slight discoloration around the crack edges, consistent with oxidation or heat stress."
        return "Based on the visual evidence, the surface shows signs of mechanical stress with localised material degradation."

    # ── Batch VQA ──────────────────────────────────────────────────────────

    def batch_vqa(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        questions: list[str] | None = None,
    ) -> list[dict]:
        """Run a list of questions against the same image."""
        if questions is None:
            questions = DEFAULT_QUESTIONS
        return [self.answer(image, q) for q in questions]

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")
