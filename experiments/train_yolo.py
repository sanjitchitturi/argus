"""
Argus Training Script — Fine-tune YOLOv8 on custom defect dataset.

Usage:
    python experiments/train_yolo.py \
        --data data/defects.yaml \
        --model yolov8n.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch 16 \
        --device cpu
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# ── allow running from project root ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8 for defect detection")
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="Base model weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--project", default="models/yolo", help="Output directory")
    p.add_argument("--name", default="argus_yolo", help="Run name")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--mlflow", action="store_true", help="Log to MLflow")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ARGUS — YOLOv8 Training")
    logger.info("=" * 60)
    logger.info(f"  Data    : {args.data}")
    logger.info(f"  Model   : {args.model}")
    logger.info(f"  Epochs  : {args.epochs}")
    logger.info(f"  Device  : {args.device}")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)

    run_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        plots=True,
        verbose=True,
    )

    if args.mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment("argus_yolo_training")
        with mlflow.start_run(run_name=args.name):
            mlflow.log_params(
                {
                    "base_model": args.model,
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "device": args.device,
                }
            )
            results = model.train(**run_kwargs)
            # Log final metrics
            if results and hasattr(results, "results_dict"):
                for k, v in results.results_dict.items():
                    try:
                        mlflow.log_metric(k.replace("/", "_"), float(v))
                    except Exception:
                        pass
    else:
        results = model.train(**run_kwargs)

    # Validate
    logger.info("Running validation on best weights...")
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        val_model = YOLO(str(best_weights))
        val_results = val_model.val(data=args.data, device=args.device)
        logger.info(f"Validation mAP@0.5 : {val_results.box.map50:.4f}")
        logger.info(f"Validation mAP@0.5:0.95 : {val_results.box.map:.4f}")
    else:
        logger.warning(f"Best weights not found at {best_weights}")

    logger.info("Training complete ✓")


if __name__ == "__main__":
    main()
