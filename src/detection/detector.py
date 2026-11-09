"""
Argus Detection Module — YOLOv8 Defect Detector
"""

from __future__ import annotations
import time
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image
from loguru import logger

try:
    import os

    if os.getenv("ARGUS_MODE", "demo") == "demo":
        raise ImportError("Explicit demo mode")
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed — using mock detector")


# ── Default defect classes (industrial inspection) ──────────────────────────
DEFECT_CLASSES = [
    "crack",
    "scratch",
    "dent",
    "corrosion",
    "hole",
    "delamination",
    "blister",
    "contamination",
    "missing_component",
    "misalignment",
]


class DefectDetector:
    """
    YOLOv8-based defect detection engine.

    Parameters
    ----------
    model_path : str
        Path to a fine-tuned YOLOv8 .pt file, or a model name like
        'yolov8n.pt' to use a pre-trained checkpoint.
    confidence : float
        Detection confidence threshold (0–1).
    iou_threshold : float
        IoU threshold for NMS.
    device : str
        'cpu', 'cuda', or 'mps'.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self._load_model()

    # ── Loading ────────────────────────────────────────────────────────────

    def _load_model(self):
        if os.getenv("ARGUS_MODE", "demo") == "demo":
            return
        if os.getenv("ARGUS_MODE") != "model":
            raise ValueError("ARGUS_MODE must be demo or model")
        if not YOLO_AVAILABLE:
            raise RuntimeError("Model dependencies missing; install requirements-models.txt")
            logger.warning("YOLO unavailable — DefectDetector running in mock mode")
            return
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLOv8 loaded from '{self.model_path}' on {self.device}")
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}")
            raise RuntimeError("Model loading failed; check dependencies and checkpoint") from exc

    # ── Inference ──────────────────────────────────────────────────────────

    def detect(self, image: Union[str, Path, np.ndarray, Image.Image]) -> dict:
        """
        Run defect detection on an image.

        Returns
        -------
        dict with keys:
            detections  : list of {class, confidence, bbox, area, severity}
            summary     : human-readable string
            inference_ms: wall-clock inference time in ms
            image_size  : (width, height)
        """
        img_array = self._to_array(image)
        h, w = img_array.shape[:2]

        t0 = time.perf_counter()

        if self.model is None:
            detections = self._mock_detections(w, h)
        else:
            detections = self._run_yolo(img_array)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        summary = self._build_summary(detections)

        return {
            "mode": "demo" if self.model is None else "model",
            "synthetic": self.model is None,
            "detections": detections,
            "summary": summary,
            "inference_ms": round(elapsed_ms, 2),
            "image_size": {"width": w, "height": h},
            "num_defects": len(detections),
        }

    def _run_yolo(self, img: np.ndarray) -> list[dict]:
        results = self.model.predict(
            source=img,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w_box = x2 - x1
                h_box = y2 - y1
                area = w_box * h_box
                cls_name = r.names.get(cls_id, f"class_{cls_id}")
                detections.append(
                    {
                        "class": cls_name,
                        "confidence": round(conf, 4),
                        "bbox": {
                            "x1": round(x1),
                            "y1": round(y1),
                            "x2": round(x2),
                            "y2": round(y2),
                            "width": round(w_box),
                            "height": round(h_box),
                        },
                        "area_px": round(area),
                        "severity": self._severity(conf, area),
                    }
                )
        return detections

    def _mock_detections(self, w: int, h: int) -> list[dict]:
        """Deterministic mock detections when YOLO is unavailable."""
        import random

        rng = random.Random(42)
        n = rng.randint(1, 3)
        dets = []
        for _ in range(n):
            cls = rng.choice(DEFECT_CLASSES)
            conf = round(rng.uniform(0.45, 0.95), 4)
            x1 = rng.randint(0, max(0, w // 2 - 1))
            y1 = rng.randint(0, max(0, h // 2 - 1))
            x2 = min(x1 + max(1, w // 4), w)
            y2 = min(y1 + max(1, h // 4), h)
            area = (x2 - x1) * (y2 - y1)
            dets.append(
                {
                    "class": cls,
                    "confidence": conf,
                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    },
                    "area_px": area,
                    "severity": self._severity(conf, area),
                    "_mock": True,
                }
            )
        return dets

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_array(image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _severity(confidence: float, area_px: float) -> str:
        if confidence > 0.80 or area_px > 15000:
            return "HIGH"
        if confidence > 0.55 or area_px > 5000:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _build_summary(detections: list[dict]) -> str:
        if not detections:
            return "No defects detected. Surface appears nominal."
        classes = [d["class"] for d in detections]
        high = sum(1 for d in detections if d["severity"] == "HIGH")
        msg = f"Detected {len(detections)} defect(s): {', '.join(classes)}."
        if high:
            msg += f" {high} HIGH-severity defect(s) require immediate attention."
        return msg

    # ── Visualisation ──────────────────────────────────────────────────────

    def draw_detections(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Return RGB image with bounding boxes drawn."""
        img = self._to_array(image).copy()
        result = self.detect(img)
        SEVERITY_COLORS = {"HIGH": (220, 38, 38), "MEDIUM": (234, 179, 8), "LOW": (34, 197, 94)}
        for det in result["detections"]:
            b = det["bbox"]
            color = SEVERITY_COLORS.get(det["severity"], (255, 255, 255))
            cv2.rectangle(img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, 2)
            label = f"{det['class']} {det['confidence']:.2f} [{det['severity']}]"
            cv2.putText(
                img, label, (b["x1"], b["y1"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )
        return img

    # ── Fine-tuning helper ─────────────────────────────────────────────────

    def train(self, data_yaml: str, epochs: int = 50, imgsz: int = 640, **kwargs):
        """Fine-tune YOLOv8 on a custom dataset."""
        if self.model is None:
            raise RuntimeError("YOLO model not loaded")
        logger.info(f"Starting YOLOv8 training: epochs={epochs}, imgsz={imgsz}")
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            device=self.device,
            **kwargs,
        )
