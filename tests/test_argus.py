"""
Argus Test Suite
Tests: detection metrics, VLM mock, LLM template, pipeline orchestration.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Allow running tests from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def sample_image():
    """640x480 blank RGB image."""
    return Image.fromarray(np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8), "RGB")


@pytest.fixture
def sample_bbox():
    return {"x1": 100, "y1": 100, "x2": 200, "y2": 200}


@pytest.fixture
def sample_detections():
    return [
        {
            "class": "crack",
            "confidence": 0.91,
            "severity": "HIGH",
            "bbox": {"x1": 50, "y1": 50, "x2": 150, "y2": 100},
            "area_px": 5000,
        },
        {
            "class": "corrosion",
            "confidence": 0.62,
            "severity": "MEDIUM",
            "bbox": {"x1": 200, "y1": 150, "x2": 280, "y2": 200},
            "area_px": 4000,
        },
    ]


# ── Detection metrics ──────────────────────────────────────────────────────


class TestIoU:
    def test_perfect_overlap(self, sample_bbox):
        from src.evaluation.metrics import compute_iou

        assert compute_iou(sample_bbox, sample_bbox) == 1.0

    def test_no_overlap(self):
        from src.evaluation.metrics import compute_iou

        a = {"x1": 0, "y1": 0, "x2": 50, "y2": 50}
        b = {"x1": 100, "y1": 100, "x2": 200, "y2": 200}
        assert compute_iou(a, b) == 0.0

    def test_partial_overlap(self):
        from src.evaluation.metrics import compute_iou

        a = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
        b = {"x1": 50, "y1": 50, "x2": 150, "y2": 150}
        iou = compute_iou(a, b)
        assert 0.0 < iou < 1.0
        # intersection = 50*50=2500, union = 10000+10000-2500=17500
        assert abs(iou - 2500 / 17500) < 1e-3

    def test_returns_float(self, sample_bbox):
        from src.evaluation.metrics import compute_iou

        result = compute_iou(sample_bbox, sample_bbox)
        assert isinstance(result, float)


class TestMAP:
    def test_perfect_detection(self, sample_detections):
        from src.evaluation.metrics import compute_map

        preds = sample_detections
        gts = sample_detections
        result = compute_map(preds, gts, iou_threshold=0.5)
        assert result["mAP"] > 0.0
        assert "per_class_AP" in result
        assert "precision" in result
        assert "recall" in result

    def test_no_detections(self, sample_detections):
        from src.evaluation.metrics import compute_map

        result = compute_map([], sample_detections)
        assert result["mAP"] == 0.0

    def test_no_ground_truth(self, sample_detections):
        from src.evaluation.metrics import compute_map

        result = compute_map(sample_detections, [])
        assert result["mAP"] == 0.0


# ── Captioning metrics ─────────────────────────────────────────────────────


class TestBLEU:
    def test_identical_sentences(self):
        from src.evaluation.metrics import compute_bleu

        hyps = ["the cat sat on the mat"]
        refs = [["the cat sat on the mat"]]
        scores = compute_bleu(hyps, refs)
        # BLEU-1 should be 1.0 for identical strings
        assert scores["BLEU-1"] >= 0.9

    def test_completely_different(self):
        from src.evaluation.metrics import compute_bleu

        hyps = ["apple banana cherry"]
        refs = [["fox jumps over fence"]]
        scores = compute_bleu(hyps, refs)
        assert scores["BLEU-1"] < 0.2

    def test_returns_all_ngrams(self):
        from src.evaluation.metrics import compute_bleu

        scores = compute_bleu(["a b c d e"], [["a b c d e"]])
        for n in range(1, 5):
            assert f"BLEU-{n}" in scores


# ── Report quality metrics ─────────────────────────────────────────────────


class TestROUGE:
    def test_identical_reports(self):
        from src.evaluation.metrics import compute_rouge

        text = "This surface has a crack defect with high severity."
        result = compute_rouge([text], [text])
        assert result["rouge1"]["fmeasure"] == pytest.approx(1.0, abs=1e-3)

    def test_returns_expected_keys(self):
        from src.evaluation.metrics import compute_rouge

        result = compute_rouge(["test"], ["test"])
        for rt in ["rouge1", "rouge2", "rougeL"]:
            assert rt in result
            assert "precision" in result[rt]
            assert "recall" in result[rt]
            assert "fmeasure" in result[rt]


# ── Detector (mock mode) ───────────────────────────────────────────────────


class TestDefectDetector:
    def test_detect_returns_dict(self, sample_image):
        from src.detection.detector import DefectDetector

        det = DefectDetector()
        result = det.detect(sample_image)
        assert isinstance(result, dict)
        assert "detections" in result
        assert "num_defects" in result
        assert "summary" in result
        assert "inference_ms" in result

    def test_detect_pil_image(self, sample_image):
        from src.detection.detector import DefectDetector

        det = DefectDetector()
        result = det.detect(sample_image)
        assert isinstance(result["detections"], list)

    def test_detect_numpy_array(self, sample_image):
        from src.detection.detector import DefectDetector

        det = DefectDetector()
        arr = np.array(sample_image)
        result = det.detect(arr)
        assert "detections" in result

    def test_severity_categories(self):
        from src.detection.detector import DefectDetector

        assert DefectDetector._severity(0.91, 20000) == "HIGH"
        assert DefectDetector._severity(0.65, 6000) == "MEDIUM"
        assert DefectDetector._severity(0.40, 1000) == "LOW"

    def test_draw_detections_returns_array(self, sample_image):
        from src.detection.detector import DefectDetector

        det = DefectDetector()
        annotated = det.draw_detections(sample_image)
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape[2] == 3


# ── VLM (mock mode) ───────────────────────────────────────────────────────


class TestVisionLanguageModel:
    def test_caption_returns_dict(self, sample_image):
        from src.vlm.vision_language import VisionLanguageModel

        vlm = VisionLanguageModel(model_name="blip2")
        result = vlm.caption(sample_image)
        assert "caption" in result
        assert isinstance(result["caption"], str)
        assert len(result["caption"]) > 0

    def test_answer_returns_dict(self, sample_image):
        from src.vlm.vision_language import VisionLanguageModel

        vlm = VisionLanguageModel()
        result = vlm.answer(sample_image, "What defect is visible?")
        assert "question" in result
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_batch_vqa_returns_list(self, sample_image):
        from src.vlm.vision_language import VisionLanguageModel

        vlm = VisionLanguageModel()
        questions = ["What is visible?", "Where is the defect?"]
        results = vlm.batch_vqa(sample_image, questions)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_mock_answers_various_questions(self, sample_image):
        from src.vlm.vision_language import VisionLanguageModel

        vlm = VisionLanguageModel()
        for q in [
            "Where is the defect located?",
            "How severe is the defect?",
            "What is the color?",
        ]:
            r = vlm.answer(sample_image, q)
            assert len(r["answer"]) > 5


# ── Report Generator ───────────────────────────────────────────────────────


class TestReportGenerator:
    def test_generate_returns_dict(self, sample_detections):
        from src.llm.report_generator import InspectionReportGenerator

        gen = InspectionReportGenerator(model_name="template")
        vqa = [{"question": "What defect?", "answer": "A crack."}]
        result = gen.generate(
            detections=sample_detections,
            caption="Metal surface with defects.",
            vqa_answers=vqa,
            image_id="TEST-001",
        )
        assert "report_text" in result
        assert "metadata" in result
        assert isinstance(result["report_text"], str)
        assert len(result["report_text"]) > 100

    def test_report_has_sections(self, sample_detections):
        from src.llm.report_generator import InspectionReportGenerator

        gen = InspectionReportGenerator(model_name="template")
        result = gen.generate(
            detections=sample_detections,
            caption="Test caption.",
            vqa_answers=[{"question": "Q", "answer": "A"}],
            image_id="TEST-002",
        )
        text = result["report_text"]
        for section in [
            "EXECUTIVE SUMMARY",
            "DEFECT ANALYSIS",
            "RISK ASSESSMENT",
            "RECOMMENDED ACTIONS",
        ]:
            assert section in text

    def test_no_defect_report(self):
        from src.llm.report_generator import InspectionReportGenerator

        gen = InspectionReportGenerator(model_name="template")
        result = gen.generate(
            detections=[],
            caption="Clean surface.",
            vqa_answers=[],
            image_id="CLEAN-001",
        )
        assert "No defects" in result["report_text"] or "nominal" in result["report_text"].lower()

    def test_metadata_fields(self, sample_detections):
        from src.llm.report_generator import InspectionReportGenerator

        gen = InspectionReportGenerator(model_name="template")
        result = gen.generate(sample_detections, "Caption", [], "META-001", "Test Inspector")
        meta = result["metadata"]
        assert meta["image_id"] == "META-001"
        assert meta["inspector"] == "Test Inspector"
        assert meta["num_defects"] == len(sample_detections)


# ── Pipeline (end-to-end) ─────────────────────────────────────────────────


class TestArgusPipeline:
    def test_full_run(self, sample_image):
        from src.pipeline import ArgusPipeline

        pipeline = ArgusPipeline()
        result = pipeline.run(sample_image, save_results=False)
        assert "image_id" in result
        assert "detection" in result
        assert "caption" in result
        assert "vqa" in result
        assert "report" in result
        assert "pipeline_metadata" in result

    def test_custom_questions(self, sample_image):
        from src.pipeline import ArgusPipeline

        pipeline = ArgusPipeline()
        questions = ["Custom question one?", "Custom question two?"]
        result = pipeline.run(sample_image, custom_questions=questions, save_results=False)
        assert len(result["vqa"]) == len(questions)

    def test_visualize_returns_array(self, sample_image):
        from src.pipeline import ArgusPipeline

        pipeline = ArgusPipeline()
        annotated = pipeline.visualize(sample_image)
        assert isinstance(annotated, np.ndarray)

    def test_severity_summary(self):
        from src.pipeline import ArgusPipeline

        dets = [
            {"severity": "HIGH"},
            {"severity": "HIGH"},
            {"severity": "MEDIUM"},
            {"severity": "LOW"},
        ]
        summary = ArgusPipeline._severity_summary(dets)
        assert summary["HIGH"] == 2
        assert summary["MEDIUM"] == 1
        assert summary["LOW"] == 1


# ── JSON output shape ─────────────────────────────────────────────────────


class TestOutputSchema:
    def test_detection_keys(self, sample_image):
        from src.detection.detector import DefectDetector

        det = DefectDetector()
        result = det.detect(sample_image)
        for key in ["detections", "summary", "inference_ms", "image_size", "num_defects"]:
            assert key in result, f"Missing key: {key}"

    def test_detection_bbox_keys(self, sample_image):
        from src.detection.detector import DefectDetector

        det = DefectDetector()
        result = det.detect(sample_image)
        for d in result["detections"]:
            for key in ["class", "confidence", "bbox", "area_px", "severity"]:
                assert key in d
            for bk in ["x1", "y1", "x2", "y2", "width", "height"]:
                assert bk in d["bbox"]

    def test_pipeline_output_serialisable(self, sample_image):
        from src.pipeline import ArgusPipeline

        pipeline = ArgusPipeline()
        result = pipeline.run(sample_image, save_results=False)
        serialised = json.dumps(result, default=str)
        recovered = json.loads(serialised)
        assert recovered["image_id"] == result["image_id"]
