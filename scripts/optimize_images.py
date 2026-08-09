#!/usr/bin/env python3
"""Optimize images for the web (resize + recompress).

Reuses Pillow (already a project dependency). Usage:

    python3 scripts/optimize_images.py

Targets:
    - Profile photo -> images/profile.jpg (max 800px, JPEG q80, progressive)
    - Article thumbnails -> images/*.png (max 800px, PNG optimized)

Extend the TARGETS list to optimize more images (e.g. article thumbnails).
"""
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
MAX_DIM = 800
JPEG_QUALITY = 80

TARGETS = [
    {
        "source": Path.home() / "Downloads" / "me.jpeg",
        "dest": ROOT / "images" / "profile.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
    {
        "source": ROOT / "images" / "laptopv0.001.png",
        "dest": ROOT / "images" / "laptopv0.001.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
    {
        "source": ROOT / "images" / "merlinv0.001.png",
        "dest": ROOT / "images" / "merlinv0.001.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
    {
        "source": ROOT / "images" / "alchemize_pytensor_mlx_gemma_3n.png",
        "dest": ROOT / "images" / "alchemize_pytensor_mlx_gemma_3n.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
    {
        "source": ROOT / "images" / "baby_steps_for_causal_discovery.png",
        "dest": ROOT / "images" / "baby_steps_for_causal_discovery.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
    {
        "source": ROOT / "images" / "bayesian_models_and_risk_optimization.png",
        "dest": ROOT / "images" / "bayesian_models_and_risk_optimization.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
    {
        "source": ROOT / "images" / "decision_making_under_contradictions.png",
        "dest": ROOT / "images" / "decision_making_under_contradictions.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
    {
        "source": ROOT / "images" / "from_experiments_to_priors.png",
        "dest": ROOT / "images" / "from_experiments_to_priors.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
    {
        "source": ROOT / "images" / "nomore_experiments_without_causality.png",
        "dest": ROOT / "images" / "nomore_experiments_without_causality.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
    {
        "source": ROOT / "images" / "placebo_bayesian_quasi_experiments.png",
        "dest": ROOT / "images" / "placebo_bayesian_quasi_experiments.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
    {
        "source": Path.home() / "Downloads" / "hours_for_article_image.png",
        "dest": ROOT / "images" / "hours_for_article_image.jpg",
        "max_dim": MAX_DIM,
        "quality": JPEG_QUALITY,
        "fmt": "JPEG",
    },
]


def optimize(source: Path, dest: Path, max_dim: int, quality: int, fmt: str = "JPEG") -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source image not found: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)  # honor orientation
        if fmt == "JPEG" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        elif fmt == "PNG" and img.mode == "RGBA":
            pass  # preserve alpha for PNG
        w, h = img.size
        scale = min(1.0, max_dim / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        if fmt == "PNG":
            img.save(dest, format="PNG", optimize=True)
        else:
            img.save(dest, format="JPEG", quality=quality, optimize=True, progressive=True)
    src_kb = source.stat().st_size / 1024
    dst_kb = dest.stat().st_size / 1024
    print(f"  {source.name} -> {dest.relative_to(ROOT)}  {src_kb:.0f}KB -> {dst_kb:.0f}KB")


def main() -> None:
    print("Optimizing images...")
    for t in TARGETS:
        optimize(**t)
    print("Done.")


if __name__ == "__main__":
    main()
