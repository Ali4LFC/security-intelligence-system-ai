"""
scripts/test_model.py
Quick diagnostic: test if OpenCV can load the YOLOv4-tiny model from models/.
Usage:  python scripts/test_model.py
"""

import os
import sys
import time
from pathlib import Path

ROOT_DIR   = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"

cfg     = MODELS_DIR / "yolov4-tiny.cfg"
weights = MODELS_DIR / "yolov4-tiny.weights"

print("OpenCV version: ", end="")
import cv2
print(cv2.__version__)

for p in (cfg, weights):
    if not p.exists():
        print(f"[!!] Missing: {p}")
        sys.exit(1)
    print(f"File: {p.name}  ({p.stat().st_size:,} bytes)")

print("\n[1/3] Loading network with readNetFromDarknet...")
t0 = time.time()
try:
    net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
    print(f"      OK — {time.time() - t0:.1f}s")
except Exception as exc:
    print(f"      FAILED: {exc}")
    sys.exit(1)

print("[2/3] Getting output layers...")
t0 = time.time()
ln  = net.getLayerNames()
out = net.getUnconnectedOutLayers()
if len(out.shape) == 2:
    out = out.flatten()
layers = [ln[i - 1] for i in out]
print(f"      OK — {layers} — {time.time() - t0:.1f}s")

print("[3/3] Running forward pass on dummy 416×416 image...")
import numpy as np
dummy = np.zeros((416, 416, 3), dtype=np.uint8)
blob  = cv2.dnn.blobFromImage(dummy, 1 / 255.0, (416, 416), swapRB=True)
net.setInput(blob)
t0 = time.time()
try:
    result = net.forward(layers)
    print(f"      OK — {time.time() - t0:.1f}s")
    print(f"      Output shapes: {[r.shape for r in result]}")
except Exception as exc:
    print(f"      FAILED: {exc}")
    sys.exit(1)

print("\nAll tests passed! Model works correctly.")
