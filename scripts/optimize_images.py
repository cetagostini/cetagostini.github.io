#!/usr/bin/env python3
"""Optimize images for the web (resize + recompress).

Reuses Pillow (already a project dependency). Usage:

    python3 scripts/optimize_images.py

Targets:
    - Profile photo -> images/profile.jpg (max 800px, JPEG q80, progressive)

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
    },
]


def optimize(source: Path, dest: Path, max_dim: int, quality: int) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source image not found: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)  # honor orientation
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_dim / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img.save(
            dest,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
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
