"""
Argus Benchmark Script
Runs the full pipeline on a folder of images, computes metrics,
and saves a consolidated evaluation report.

Usage:
    python experiments/benchmark.py \
        --images data/samples \
        --ground_truth data/ground_truth.json \
        --output experiments/benchmark_results.json
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from PIL import Image

from src.pipeline import ArgusPipeline
from src.evaluation.metrics import run_full_evaluation
from src.utils.config import save_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Argus Benchmark Runner")
    p.add_argument("--images", required=True, help="Folder of test images")
    p.add_argument("--ground_truth", default=None, help="Ground-truth JSON file")
    p.add_argument("--output", default="benchmark_results.json")
    p.add_argument("--yolo", default="yolov8n.pt")
    p.add_argument("--vlm", default="blip2")
    p.add_argument("--llm", default="template")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_images", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ARGUS — Benchmark Runner")
    logger.info("=" * 60)

    # Collect image paths
    img_dir = Path(args.images)
    if not img_dir.exists():
        logger.error(f"Image directory not found: {img_dir}")
        sys.exit(1)

    img_paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.max_images:
        img_paths = img_paths[: args.max_images]

    if not img_paths:
        logger.error("No images found")
        sys.exit(1)

    logger.info(f"Found {len(img_paths)} images")

    # Load pipeline
    logger.info("Loading Argus pipeline...")
    pipeline = ArgusPipeline(
        yolo_model=args.yolo,
        vlm_model=args.vlm,
        llm_model=args.llm,
        device=args.device,
    )

    # Load ground truth if provided
    ground_truths = []
    ref_captions = []
    ref_reports = []
    if args.ground_truth and Path(args.ground_truth).exists():
        with open(args.ground_truth) as f:
            gt_data = json.load(f)
        ground_truths = gt_data.get("detections", [])
        ref_captions = gt_data.get("captions", [])
        ref_reports = gt_data.get("reports", [])
        logger.info(f"Ground truth loaded: {len(ground_truths)} annotations")

    # ── Run inference ───────────────────────────────────────────────────
    all_results = []
    all_preds = []
    all_captions = []
    all_reports = []
    total_defects = 0
    t_wall = time.perf_counter()

    for i, img_path in enumerate(img_paths, 1):
        logger.info(f"[{i}/{len(img_paths)}] Processing: {img_path.name}")
        try:
            img = Image.open(img_path).convert("RGB")
            result = pipeline.run(image=img, image_id=img_path.stem, save_results=False)
            all_results.append(result)
            all_preds.append(result["detection"])
            all_captions.append(result["caption"]["caption"])
            all_reports.append(result["report"]["report_text"])
            total_defects += result["detection"]["num_defects"]
        except Exception as exc:
            logger.error(f"Failed on {img_path.name}: {exc}")

    elapsed = time.perf_counter() - t_wall
    avg_ms = (elapsed / len(all_results) * 1000) if all_results else 0

    logger.info(f"Inference done in {elapsed:.1f}s — avg {avg_ms:.0f}ms/image")

    # ── Compute metrics ─────────────────────────────────────────────────
    metrics = {}
    if ground_truths:
        logger.info("Computing evaluation metrics...")
        metrics = run_full_evaluation(
            predictions=all_preds,
            ground_truths=ground_truths,
            captions=all_captions,
            reference_captions=[[c] for c in ref_captions]
            if ref_captions
            else [[""] for _ in all_captions],
            reports=all_reports,
            reference_reports=ref_reports if ref_reports else ["" for _ in all_reports],
        )

    # ── Assemble report ─────────────────────────────────────────────────
    benchmark = {
        "summary": {
            "total_images": len(img_paths),
            "processed": len(all_results),
            "total_defects_found": total_defects,
            "avg_defects_per_image": round(total_defects / max(len(all_results), 1), 2),
            "total_wall_time_s": round(elapsed, 2),
            "avg_inference_ms": round(avg_ms, 2),
        },
        "config": {
            "yolo_model": args.yolo,
            "vlm_model": args.vlm,
            "llm_model": args.llm,
            "device": args.device,
        },
        "metrics": metrics,
        "per_image": [
            {
                "image_id": r["image_id"],
                "num_defects": r["detection"]["num_defects"],
                "detection_ms": r["detection"]["inference_ms"],
                "caption_ms": r["caption"]["inference_ms"],
                "severity_breakdown": r["pipeline_metadata"]["severity_breakdown"],
            }
            for r in all_results
        ],
    }

    save_json(benchmark, args.output)

    logger.info("=" * 60)
    logger.info(f"Benchmark complete — results → {args.output}")
    logger.info(f"Images processed : {len(all_results)}")
    logger.info(f"Total defects    : {total_defects}")
    logger.info(f"Avg time/image   : {avg_ms:.0f} ms")
    if metrics:
        logger.info(f"mAP              : {metrics.get('detection', {}).get('mAP', 'N/A')}")
        logger.info(f"BLEU-4           : {metrics.get('captioning', {}).get('BLEU-4', 'N/A')}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
