"""
Argus Evaluation Module
Metrics: IoU, mAP (detection) | BLEU (captioning) | ROUGE (report quality)
"""

from __future__ import annotations
import json
import math
from numbers import Real
from typing import List, Dict, Any

import numpy as np
from loguru import logger

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


# ══════════════════════════════════════════════════════════
# Detection Metrics
# ══════════════════════════════════════════════════════════


def _validated_box(box: dict) -> tuple[float, float, float, float]:
    if not isinstance(box, dict):
        raise ValueError("Bounding box must be an object")
    try:
        values = tuple(box[name] for name in ("x1", "y1", "x2", "y2"))
    except KeyError as exc:
        raise ValueError(f"Bounding box is missing {exc.args[0]}") from exc
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("Bounding box coordinates must be finite numbers")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box must have positive width and height")
    return float(x1), float(y1), float(x2), float(y2)


def compute_iou(box_a: dict, box_b: dict) -> float:
    """
    Compute Intersection over Union between two bounding boxes.

    Parameters
    ----------
    box_a, box_b : dict with keys x1, y1, x2, y2
    """
    ax1, ay1, ax2, ay2 = _validated_box(box_a)
    bx1, by1, bx2, by2 = _validated_box(box_b)
    xa = max(ax1, bx1)
    ya = max(ay1, by1)
    xb = min(ax2, bx2)
    yb = min(ay2, by2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union_area = area_a + area_b - inter_area
    if union_area == 0:
        return 0.0
    return round(inter_area / union_area, 4)


def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    class_names: List[str] | None = None,
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP@IoU) for detection.

    Parameters
    ----------
    predictions : list of {class, confidence, bbox}
    ground_truths : list of {class, bbox}
    iou_threshold : IoU threshold for TP/FP
    class_names : optional explicit class list

    Returns
    -------
    dict: {mAP, per_class_AP, precision, recall}
    """
    if (
        isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, Real)
        or not math.isfinite(iou_threshold)
        or not 0 < iou_threshold <= 1
    ):
        raise ValueError("IoU threshold must be in (0, 1]")
    for detection in predictions + ground_truths:
        if not isinstance(detection, dict):
            raise ValueError("Detection must be an object")
        if not isinstance(detection.get("class"), str) or not detection["class"].strip():
            raise ValueError("Detection class must be nonempty text")
        _validated_box(detection.get("bbox"))
    for prediction in predictions:
        confidence = prediction.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, Real)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Prediction confidence must be in [0, 1]")
    if not class_names:
        class_names = sorted({d["class"] for d in ground_truths})

    per_class_ap = {}
    total_tp = 0

    for cls in class_names:
        preds = sorted(
            [p for p in predictions if p["class"] == cls],
            key=lambda x: x["confidence"],
            reverse=True,
        )
        gts = [g for g in ground_truths if g["class"] == cls]

        if not gts:
            per_class_ap[cls] = 0.0
            continue
        if not preds:
            per_class_ap[cls] = 0.0
            continue

        matched = [False] * len(gts)
        tps, fps = [], []

        for pred in preds:
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(gts):
                if matched[i] or pred.get("image_id") != gt.get("image_id"):
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_idx >= 0 and best_iou >= iou_threshold:
                tps.append(1)
                fps.append(0)
                matched[best_idx] = True
            else:
                tps.append(0)
                fps.append(1)

        total_tp += sum(tps)
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
        recalls = tp_cum / len(gts)

        # Area under PR curve (11-point interpolation)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            prec_at_rec = precisions[recalls >= t]
            ap += np.max(prec_at_rec) if len(prec_at_rec) else 0.0
        per_class_ap[cls] = round(ap / 11, 4)

    map_score = round(float(np.mean(list(per_class_ap.values()))), 4) if per_class_ap else 0.0

    # Overall precision / recall
    precision = round(total_tp / max(len(predictions), 1), 4)
    recall = round(total_tp / max(len(ground_truths), 1), 4)

    return {
        "mAP": map_score,
        "per_class_AP": per_class_ap,
        "precision": precision,
        "recall": recall,
        "iou_threshold": iou_threshold,
    }


# ══════════════════════════════════════════════════════════
# Captioning Metrics
# ══════════════════════════════════════════════════════════


def compute_bleu(
    hypotheses: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """
    Compute BLEU-1 through BLEU-4 scores.

    Parameters
    ----------
    hypotheses : list of generated captions
    references : list of lists of reference captions (one list per image)
    """
    if not NLTK_AVAILABLE:
        raise RuntimeError("nltk is required to calculate BLEU")
    if len(hypotheses) != len(references) or any(not refs for refs in references):
        raise ValueError("Each caption requires at least one aligned reference")
    if not hypotheses:
        return {}

    smooth = SmoothingFunction().method1
    tokenized_hyps = [h.lower().split() for h in hypotheses]
    tokenized_refs = [[r.lower().split() for r in refs] for refs in references]

    scores = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
        score = corpus_bleu(
            tokenized_refs, tokenized_hyps, weights=weights, smoothing_function=smooth
        )
        scores[f"BLEU-{n}"] = round(score, 4)
    return scores


def compute_bleu_single(hypothesis: str, references: List[str]) -> Dict[str, float]:
    """Compute BLEU for a single hypothesis."""
    return compute_bleu([hypothesis], [references])


# ══════════════════════════════════════════════════════════
# Report Quality Metrics
# ══════════════════════════════════════════════════════════


def compute_rouge(
    hypotheses: List[str],
    references: List[str],
    rouge_types: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE scores for generated reports.

    Parameters
    ----------
    hypotheses : generated reports
    references : reference / gold reports
    rouge_types : ['rouge1', 'rouge2', 'rougeL']

    Returns
    -------
    dict: {rouge_type: {precision, recall, fmeasure}}
    """
    if not ROUGE_AVAILABLE:
        raise RuntimeError("rouge-score is required to calculate ROUGE")
    if len(hypotheses) != len(references):
        raise ValueError("Reports and references must have equal length")
    if not hypotheses:
        return {}

    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL"]

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    aggregated: Dict[str, Dict[str, list]] = {
        r: {"precision": [], "recall": [], "fmeasure": []} for r in rouge_types
    }

    for hyp, ref in zip(hypotheses, references):
        result = scorer.score(ref, hyp)
        for rt in rouge_types:
            aggregated[rt]["precision"].append(result[rt].precision)
            aggregated[rt]["recall"].append(result[rt].recall)
            aggregated[rt]["fmeasure"].append(result[rt].fmeasure)

    return {
        rt: {
            "precision": round(float(np.mean(vals["precision"])), 4),
            "recall": round(float(np.mean(vals["recall"])), 4),
            "fmeasure": round(float(np.mean(vals["fmeasure"])), 4),
        }
        for rt, vals in aggregated.items()
    }


# ══════════════════════════════════════════════════════════
# Full Evaluation Pipeline
# ══════════════════════════════════════════════════════════


def run_full_evaluation(
    predictions: List[Dict],
    ground_truths: List[Dict],
    captions: List[str],
    reference_captions: List[List[str]],
    reports: List[str],
    reference_reports: List[str],
) -> Dict[str, Any]:
    """
    Run all metrics in one call. Returns a consolidated evaluation dict.
    """
    logger.info("Running full evaluation pipeline...")

    def with_image_ids(records):
        flattened = []
        for index, record in enumerate(records):
            image_id = record.get("image_id", str(index))
            for detection in record.get("detections", []):
                flattened.append({**detection, "image_id": detection.get("image_id", image_id)})
        return flattened

    # Detection
    detection_metrics = compute_map(
        with_image_ids(predictions),
        with_image_ids(ground_truths),
    )

    # Captioning
    bleu_scores = compute_bleu(captions, reference_captions)

    # Report quality
    rouge_scores = compute_rouge(reports, reference_reports)

    result = {
        "detection": detection_metrics,
        "captioning": bleu_scores,
        "report_quality": rouge_scores,
    }

    logger.info(
        f"Evaluation complete — mAP={detection_metrics['mAP']:.4f} | "
        f"BLEU-4={bleu_scores.get('BLEU-4', 0):.4f} | "
        f"ROUGE-L={rouge_scores.get('rougeL', {}).get('fmeasure', 0):.4f}"
    )
    return result


# ══════════════════════════════════════════════════════════
# CLI helper
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Argus Evaluation CLI")
    parser.add_argument("--predictions", required=True, help="JSON file with model predictions")
    parser.add_argument("--ground_truths", required=True, help="JSON file with ground truth")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    with open(args.predictions) as f:
        preds = json.load(f)
    with open(args.ground_truths) as f:
        gts = json.load(f)

    metrics = run_full_evaluation(
        predictions=preds.get("predictions", []),
        ground_truths=gts.get("ground_truths", []),
        captions=preds.get("captions", []),
        reference_captions=gts.get("reference_captions", [[]]),
        reports=preds.get("reports", []),
        reference_reports=gts.get("reference_reports", [""]),
    )

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation results saved → {args.output}")
    print(json.dumps(metrics, indent=2))
