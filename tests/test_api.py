from io import BytesIO
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from api.main import app
from src.evaluation.metrics import compute_map, run_full_evaluation


def image_file():
    out = BytesIO()
    Image.new("RGB", (32, 32)).save(out, format="PNG")
    return ("test.png", out.getvalue(), "image/png")


def test_demo_api_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGUS_MODE", "demo")
    import src.pipeline as pipeline

    monkeypatch.setattr(pipeline, "EXPERIMENTS_DIR", tmp_path)
    with TestClient(app) as client:
        assert client.get("/health").json()["synthetic"] is True
        result = client.post(
            "/inspect", files={"file": image_file()}, data={"save_results": "false"}
        )
        assert result.status_code == 200, result.text
        assert result.json()["pipeline_metadata"]["synthetic"] is True
        assert (
            client.post(
                "/inspect", files={"file": image_file()}, data={"image_id": ".."}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/inspect", files={"file": image_file()}, data={"questions": "{}"}
            ).status_code
            == 422
        )
        assert client.post("/detect", files={"file": ("bad.png", b"bad")}).status_code == 400


def test_duplicate_boxes_do_not_inflate_recall():
    gt = {"class": "crack", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}
    pred = dict(gt, confidence=0.9)
    metrics = compute_map([pred, pred], [gt])
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 0.5
    assert compute_map([dict(pred, image_id="a")], [dict(gt, image_id="b")])["recall"] == 0


def test_map_averages_over_ground_truth_classes_only():
    gt = {"class": "crack", "image_id": "one", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}
    predictions = [
        {**gt, "confidence": 0.9},
        {"class": "scratch", "image_id": "one", "confidence": 0.8, "bbox": gt["bbox"]},
    ]

    result = compute_map(predictions, [gt])

    assert result["mAP"] == result["per_class_AP"]["crack"]
    assert "scratch" not in result["per_class_AP"]
    assert result["precision"] == 0.5


def test_evaluation_rejects_missing_text_references():
    with TestClient(app) as client:
        response = client.post("/evaluate", json={"captions": ["a crack is visible"]})
    assert response.status_code == 422


def test_full_evaluation_keeps_detections_within_images():
    box = {"class": "crack", "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}
    predictions = [{"image_id": "other", "detections": [{**box, "confidence": 0.9}]}]
    labels = [{"image_id": "target", "detections": [box]}]
    result = run_full_evaluation(predictions, labels, [], [], [], [])
    assert result["detection"]["recall"] == 0


@pytest.mark.parametrize(
    "bad_box",
    [
        {"x1": 10, "y1": 0, "x2": 0, "y2": 10},
        {"x1": 0, "y1": 0, "x2": 0, "y2": 10},
        {"x1": 0, "y1": 0, "x2": float("nan"), "y2": 10},
    ],
)
def test_detection_metrics_reject_malformed_boxes(bad_box):
    with pytest.raises(ValueError, match="Bounding box"):
        compute_map([], [{"class": "crack", "bbox": bad_box}])


def test_evaluation_api_rejects_invalid_detection_confidence():
    with TestClient(app) as client:
        response = client.post(
            "/evaluate",
            json={
                "predictions": [
                    {
                        "image_id": "one",
                        "detections": [
                            {
                                "class": "crack",
                                "confidence": 1.5,
                                "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                            }
                        ],
                    }
                ]
            },
        )
    assert response.status_code == 422
