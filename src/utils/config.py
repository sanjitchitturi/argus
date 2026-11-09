"""
Argus Utilities — Configuration & Logging
"""

import os
import yaml
import json
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
EXPERIMENTS_DIR = Path(os.getenv("ARGUS_RESULTS_DIR", str(ROOT_DIR / "outputs" / "inspections")))


def load_config(config_path: str = None) -> dict:
    """Load YAML config file."""
    if config_path is None:
        config_path = ROOT_DIR / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(log_level: str = "INFO", log_file: str = None):
    """Configure loguru logger."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> — {message}",
        colorize=True,
    )
    if log_file:
        logger.add(log_file, level=log_level, rotation="10 MB")
    return logger


def validate_image_id(value: str) -> str:
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("image_id must contain 1–80 letters, digits, underscores or hyphens")
    return value


def save_json(data: dict, path: str):
    """Save dict to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON → {path}")


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dirs():
    """Create all required directories."""
    for d in [
        DATA_DIR / "raw",
        DATA_DIR / "processed",
        DATA_DIR / "samples",
        MODELS_DIR / "yolo",
        MODELS_DIR / "vlm",
        MODELS_DIR / "llm",
        EXPERIMENTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
