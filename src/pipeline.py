"""
Argus Core Pipeline
Orchestrates: Detection → VLM → Report Generation → JSON Output
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image
from loguru import logger

from src.detection.detector import DefectDetector
from src.vlm.vision_language import VisionLanguageModel, DEFAULT_QUESTIONS
from src.llm.report_generator import InspectionReportGenerator
from src.utils.config import save_json, EXPERIMENTS_DIR, validate_image_id


class ArgusPipeline:
    """
    End-to-end Argus inspection pipeline.

    Parameters
    ----------
    yolo_model : str
        YOLOv8 model path or name.
    vlm_model : str
        VLM model name ('blip2', 'llava', or HF model ID).
    llm_model : str
        LLM model name ('mistral', 'llama', 'template', or HF model ID).
    device : str
        Compute device.
    detection_confidence : float
        YOLOv8 detection confidence threshold.
    """

    def __init__(
        self,
        yolo_model: str = "yolov8n.pt",
        vlm_model: str = "blip2",
        llm_model: str = "template",
        device: str = "cpu",
        detection_confidence: float = 0.35,
    ):
        logger.info("Initialising Argus pipeline...")
        import os

        if os.getenv("ARGUS_MODE", "demo") == "model" and not Path(yolo_model).is_file():
            raise ValueError(
                "Model mode requires a local defect-trained checkpoint, not generic COCO weights"
            )
        self.detector = DefectDetector(
            model_path=yolo_model,
            confidence=detection_confidence,
            device=device,
        )
        self.vlm = VisionLanguageModel(model_name=vlm_model, device=device)
        self.reporter = InspectionReportGenerator(model_name=llm_model, device=device)
        logger.info("Argus pipeline ready ✓")

    # ── Full Inference ─────────────────────────────────────────────────────

    def run(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        custom_questions: list[str] | None = None,
        image_id: str | None = None,
        inspector: str = "Argus AI",
        save_results: bool = True,
    ) -> dict:
        """
        Run full inspection pipeline on one image.

        Returns
        -------
        dict with keys:
            image_id, timestamp, detection, caption, vqa, report, metadata
        """
        if image_id is None:
            image_id = f"ARGUS-{uuid.uuid4().hex[:8].upper()}"

        validate_image_id(image_id)
        questions = DEFAULT_QUESTIONS if custom_questions is None else custom_questions
        if (
            not isinstance(questions, list)
            or len(questions) > 10
            or any(not isinstance(q, str) or not q.strip() or len(q) > 500 for q in questions)
        ):
            raise ValueError("Provide at most 10 nonempty questions, each up to 500 characters")

        logger.info(f"[{image_id}] Running Argus pipeline...")

        # ── Step 1: Detection ───────────────────────────────────────────
        logger.info(f"[{image_id}] Step 1/3 — Defect detection")
        detection_result = self.detector.detect(image)

        # ── Step 2: VLM — Caption + VQA ────────────────────────────────
        logger.info(f"[{image_id}] Step 2/3 — Vision-language analysis")
        caption_result = self.vlm.caption(image)
        vqa_results = self.vlm.batch_vqa(image, questions)

        # ── Step 3: Report Generation ───────────────────────────────────
        logger.info(f"[{image_id}] Step 3/3 — Generating inspection report")
        report_result = self.reporter.generate(
            detections=detection_result["detections"],
            caption=caption_result["caption"],
            vqa_answers=vqa_results,
            image_id=image_id,
            inspector=inspector,
        )

        # ── Assemble Output ─────────────────────────────────────────────
        output = {
            "image_id": image_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detection": detection_result,
            "caption": caption_result,
            "vqa": vqa_results,
            "report": report_result,
            "pipeline_metadata": {
                "mode": "demo" if self.detector.model is None else "model",
                "synthetic": self.detector.model is None,
                "notice": "Synthetic demo outputs are not image-derived findings."
                if self.detector.model is None
                else "Unvalidated model output; human review required.",
                "yolo_model": self.detector.model_path,
                "vlm_model": self.vlm.model_id,
                "llm_model": self.reporter.model_id
                if not self.reporter.use_template
                else "template",
                "device": self.detector.device,
                "total_defects": detection_result["num_defects"],
                "severity_breakdown": self._severity_summary(detection_result["detections"]),
            },
        }

        if save_results:
            out_path = EXPERIMENTS_DIR / f"{image_id}.json"
            save_json(output, str(out_path))

        logger.info(
            f"[{image_id}] Pipeline complete — {detection_result['num_defects']} defect(s) detected"
        )
        return output

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _severity_summary(detections: list[dict]) -> dict:
        counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for d in detections:
            counts[d.get("severity", "LOW")] += 1
        return counts

    def visualize(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> np.ndarray:
        """Return annotated image with detection bounding boxes."""
        return self.detector.draw_detections(image)
