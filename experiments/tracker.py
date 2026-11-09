"""
Argus MLflow Experiment Tracker
Logs detection, VLM, and report metrics to MLflow.
"""

from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from loguru import logger

try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow not installed — tracking disabled")


class ArgusTracker:
    """
    Thin wrapper around MLflow for Argus experiment tracking.

    Parameters
    ----------
    tracking_uri : str
        Path or URI for MLflow tracking server.
    experiment_name : str
        MLflow experiment name.
    """

    def __init__(
        self,
        tracking_uri: str = "./mlflow_runs",
        experiment_name: str = "argus_inspections",
    ):
        self.enabled = MLFLOW_AVAILABLE
        if not self.enabled:
            return

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking → {tracking_uri} / {experiment_name}")

    def log_inspection(
        self,
        result: dict,
        run_name: str | None = None,
        tags: dict | None = None,
    ) -> str | None:
        """
        Log a full pipeline result to MLflow.

        Parameters
        ----------
        result   : output from ArgusPipeline.run()
        run_name : optional display name for the run
        tags     : extra key-value tags

        Returns
        -------
        run_id if logging succeeded, else None
        """
        if not self.enabled:
            logger.warning("MLflow unavailable — skipping log")
            return None

        det = result.get("detection", {})
        cap = result.get("caption", {})
        report_meta = result.get("report", {}).get("metadata", {})
        pipe_meta = result.get("pipeline_metadata", {})

        with mlflow.start_run(run_name=run_name or result.get("image_id")):
            # Tags
            mlflow.set_tag("image_id", result.get("image_id", "unknown"))
            mlflow.set_tag("inspector", report_meta.get("inspector", "Argus AI"))
            mlflow.set_tag("vlm_model", pipe_meta.get("vlm_model", ""))
            mlflow.set_tag("yolo_model", pipe_meta.get("yolo_model", ""))
            mlflow.set_tag("llm_model", pipe_meta.get("llm_model", ""))
            if tags:
                for k, v in tags.items():
                    mlflow.set_tag(k, str(v))

            # Parameters
            mlflow.log_param("device", pipe_meta.get("device", "cpu"))
            mlflow.log_param("image_width", det.get("image_size", {}).get("width", 0))
            mlflow.log_param("image_height", det.get("image_size", {}).get("height", 0))

            # Metrics
            mlflow.log_metric("num_defects", det.get("num_defects", 0))
            mlflow.log_metric("detection_ms", det.get("inference_ms", 0))
            mlflow.log_metric("caption_ms", cap.get("inference_ms", 0))
            mlflow.log_metric("report_ms", report_meta.get("inference_ms", 0))

            sev = pipe_meta.get("severity_breakdown", {})
            mlflow.log_metric("high_severity", sev.get("HIGH", 0))
            mlflow.log_metric("medium_severity", sev.get("MEDIUM", 0))
            mlflow.log_metric("low_severity", sev.get("LOW", 0))

            # Artifacts — save JSON result
            with TemporaryDirectory() as directory:
                tmp = Path(directory) / "inspection.json"
                with tmp.open("w", encoding="utf-8") as handle:
                    json.dump(result, handle, indent=2, default=str)
                mlflow.log_artifact(str(tmp), artifact_path="results")

            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run logged: {run_id}")
            return run_id

    def log_metrics(self, metrics: dict, run_name: str = "evaluation") -> str | None:
        """Log evaluation metrics to MLflow."""
        if not self.enabled:
            return None

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("run_type", "evaluation")

            det = metrics.get("detection", {})
            bleu = metrics.get("captioning", {})
            rouge = metrics.get("report_quality", {})

            mlflow.log_metric("mAP", det.get("mAP", 0))
            mlflow.log_metric("precision", det.get("precision", 0))
            mlflow.log_metric("recall", det.get("recall", 0))

            for k, v in bleu.items():
                mlflow.log_metric(k.lower().replace("-", "_"), v)

            for rt, scores in rouge.items():
                for metric, val in scores.items():
                    mlflow.log_metric(f"{rt}_{metric}", val)

            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow evaluation run: {run_id}")
            return run_id

    def list_runs(self, max_results: int = 20) -> list[dict]:
        """List recent MLflow runs."""
        if not self.enabled:
            return []
        runs = mlflow.search_runs(max_results=max_results, output_format="list")
        return [
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "metrics": r.data.metrics,
                "tags": r.data.tags,
            }
            for r in runs
        ]
