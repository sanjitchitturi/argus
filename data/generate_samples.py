"""
Argus Sample Data Generator
Creates synthetic industrial defect images for testing
without a real dataset.

Usage:
    python data/generate_samples.py --n 20 --output data/samples
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


DEFECT_TYPES = [
    "crack",
    "scratch",
    "dent",
    "corrosion",
    "hole",
    "delamination",
    "blister",
    "contamination",
]

SURFACE_COLORS = [
    (180, 180, 180),  # bare metal
    (160, 140, 100),  # rusty metal
    (200, 190, 170),  # painted surface
    (100, 120, 140),  # anodized aluminium
]


def generate_surface(width: int, height: int, rng: random.Random) -> Image.Image:
    base_color = rng.choice(SURFACE_COLORS)
    noise = np.random.default_rng(rng.randint(0, 9999)).integers(
        -20, 20, (height, width, 3), dtype=np.int16
    )
    arr = np.clip(np.array(base_color, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


def draw_defect(
    draw: ImageDraw.ImageDraw, defect_type: str, bbox: tuple, rng: random.Random
) -> None:
    x1, y1, x2, y2 = bbox
    _cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    _w, _h = x2 - x1, y2 - y1

    if defect_type == "crack":
        color = (30, 20, 15)
        pts = [(x1, y1)]
        x, y = x1, y1
        for _ in range(rng.randint(4, 8)):
            x += rng.randint(10, 30)
            y += rng.randint(-15, 15)
            pts.append((min(x, x2), max(y1, min(y, y2))))
        draw.line(pts, fill=color, width=rng.randint(2, 4))

    elif defect_type == "scratch":
        color = (60, 50, 40)
        draw.line([(x1, cy), (x2, cy + rng.randint(-5, 5))], fill=color, width=rng.randint(1, 3))

    elif defect_type == "dent":
        color = (100, 95, 90)
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=(60, 55, 50), width=2)

    elif defect_type == "corrosion":
        for _ in range(rng.randint(5, 12)):
            rx = rng.randint(x1, x2 - 10)
            ry = rng.randint(y1, y2 - 10)
            rw = rng.randint(5, 20)
            rh = rng.randint(5, 20)
            color = (rng.randint(140, 180), rng.randint(70, 100), rng.randint(20, 50))
            draw.ellipse([rx, ry, rx + rw, ry + rh], fill=color)

    elif defect_type == "hole":
        color = (10, 10, 10)
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=(40, 35, 30), width=3)

    elif defect_type == "blister":
        color = (210, 200, 185)
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=(170, 160, 140), width=2)

    elif defect_type == "delamination":
        color = (220, 215, 200)
        for i in range(3):
            offset = i * 4
            draw.rectangle(
                [x1 + offset, y1 + offset, x2 - offset, y2 - offset],
                outline=(150, 140, 120),
                width=1,
            )

    elif defect_type == "contamination":
        for _ in range(rng.randint(3, 8)):
            rx = rng.randint(x1, x2)
            ry = rng.randint(y1, y2)
            r = rng.randint(3, 10)
            color = (rng.randint(10, 50), rng.randint(10, 50), rng.randint(10, 50))
            draw.ellipse([rx - r, ry - r, rx + r, ry + r], fill=color)


def generate_sample(
    idx: int,
    output_dir: Path,
    width: int = 640,
    height: int = 480,
    rng: random.Random | None = None,
) -> dict:
    if rng is None:
        rng = random.Random(idx)

    img = generate_surface(width, height, rng)
    draw = ImageDraw.Draw(img)

    n_defects = rng.randint(0, 3)
    annotations = []

    for _ in range(n_defects):
        defect = rng.choice(DEFECT_TYPES)
        x1 = rng.randint(20, width - 100)
        y1 = rng.randint(20, height - 80)
        x2 = min(x1 + rng.randint(30, 120), width - 10)
        y2 = min(y1 + rng.randint(20, 80), height - 10)
        draw_defect(draw, defect, (x1, y1, x2, y2), rng)
        area = (x2 - x1) * (y2 - y1)
        conf = rng.uniform(0.5, 0.98)
        sev = "HIGH" if conf > 0.80 or area > 10000 else ("MEDIUM" if conf > 0.55 else "LOW")
        annotations.append(
            {
                "class": defect,
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                },
                "area_px": area,
                "severity": sev,
            }
        )

    img_name = f"sample_{idx:04d}.jpg"
    img.save(output_dir / img_name, quality=90)

    return {
        "image_id": f"SAMPLE-{idx:04d}",
        "file": img_name,
        "size": {"width": width, "height": height},
        "detections": annotations,
        "num_defects": len(annotations),
        "caption": f"Industrial surface sample with {len(annotations)} visible defect(s).",
    }


def main():
    p = argparse.ArgumentParser(description="Generate synthetic defect images")
    p.add_argument("--n", type=int, default=20, help="Number of images")
    p.add_argument("--output", default="data/samples", help="Output directory")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    samples = []

    print(f"Generating {args.n} synthetic defect images → {out}")
    for i in range(args.n):
        sample_rng = random.Random(rng.randint(0, 999999))
        meta = generate_sample(i, out, args.width, args.height, sample_rng)
        samples.append(meta)
        print(f"  [{i + 1}/{args.n}] {meta['file']} — {meta['num_defects']} defect(s)")

    # Save ground truth manifest
    gt = {
        "detections": [{"detections": s["detections"]} for s in samples],
        "captions": [s["caption"] for s in samples],
        "reports": ["" for _ in samples],
        "samples": samples,
    }
    gt_path = out / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"\nDone. Ground truth → {gt_path}")


if __name__ == "__main__":
    main()
