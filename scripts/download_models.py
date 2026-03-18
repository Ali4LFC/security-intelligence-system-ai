"""
scripts/download_models.py
Download YOLOv4-tiny model files into the models/ directory.
Run once before first use:  python scripts/download_models.py
"""

import urllib.request
import os
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FILES = {
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
    "yolov4-tiny.cfg":     "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
}


def download_file(url: str, filepath: Path) -> None:
    if filepath.exists():
        print(f"  [OK] {filepath.name} already exists, skipping.")
        return

    print(f"  [..] Downloading {filepath.name}...")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb_d = downloaded / (1024 * 1024)
            mb_t = total_size / (1024 * 1024)
            sys.stdout.write(f"\r      {pct:5.1f}% ({mb_d:.1f}/{mb_t:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, filepath, reporthook=progress)
        print(f"\n  [OK] {filepath.name} downloaded successfully!")
    except Exception as exc:
        print(f"\n  [!!] Error downloading {filepath.name}: {exc}")
        if filepath.exists():
            filepath.unlink()
        raise


def main():
    print("=" * 60)
    print("  DOWNLOADING YOLOv4-tiny MODELS → models/")
    print("=" * 60)

    for filename, url in FILES.items():
        download_file(url, MODELS_DIR / filename)

    print()
    print("=" * 60)
    print("  ALL FILES DOWNLOADED SUCCESSFULLY!")
    print("  You can now run:  python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
