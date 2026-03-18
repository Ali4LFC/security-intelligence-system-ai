# Security Monitoring System — Dangerous Object Detection & People Counting

> **Project topic:** Building a program that automatically finds and counts objects in images for quality control (danger alerts). Detection of dangerous objects such as cold weapons. Counting the number of people.

---

## Overview

This application uses **computer vision (OpenCV)** and the **YOLOv4-tiny neural network** to automatically detect:

- **People** — counts the number of people in an image
- **Knives** — cold weapon detection
- **Scissors** — cutting tool detection
- **Baseball bats** — blunt weapon detection

When dangerous objects are found, the system generates a **visual threat alert**.

---

## Project Structure

```
project/
├── main.py              # Main GUI application (tkinter)
├── detector.py          # Detection module (OpenCV DNN + YOLO)
├── download_models.py   # Model download script
├── coco.names           # COCO class names (80 classes)
├── yolov4-tiny.cfg      # Neural network config (downloaded)
├── yolov4-tiny.weights  # Neural network weights (downloaded)
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

---

## Technologies

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Programming language |
| **OpenCV (cv2.dnn)** | Neural network loading and inference |
| **YOLOv4-tiny** | Object detection neural network |
| **NumPy** | Numerical array operations |
| **Tkinter / CustomTkinter** | Modern Dark GUI |
| **Pillow** | Image conversion for GUI display |

---

## Installation & Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download neural network models

```bash
python download_models.py
```

This downloads:
- `yolov4-tiny.cfg` — network configuration (~3 KB)
- `yolov4-tiny.weights` — network weights (~24 MB)

### 3. Run the application

```bash
python main.py
```

---

## How It Works

### Algorithm

1. **Image loading** → BGR format via OpenCV
2. **Preprocessing** → normalization (0–1), blob conversion (416×416)
3. **Neural network forward pass** → YOLOv4-tiny processes the blob
4. **Filtering** → detections filtered by confidence threshold
5. **NMS (Non-Maximum Suppression)** → removal of duplicate bounding boxes
6. **Threat classification** → class checking (knife, scissors, bat)
7. **Visualization** → drawing bounding boxes, labels, alert panel
8. **Warm-up** → dummy forward pass on startup to avoid first-inference delay

### Detection Parameters

| Parameter | Default | Description |
|---|---|---|
| Confidence threshold | 35% (adjustable) | Minimum detection confidence |
| NMS threshold | 40% | Overlapping box suppression |
| Input size | 416×416 | Blob size for neural network |

---

## Interface

### Controls

| Button | Action |
|---|---|
| Load Image | Select and analyze a single image |
| Batch Analysis | Analyze all images in a folder |
| Camera ON/OFF | Toggle webcam with real-time detection |
| Save Result | Save the annotated image |
| Export Report | Save a text report |

### Statistics Panel

- **People detected** — number of people in the image
- **Dangerous objects** — number of threats (knives, scissors, bats)
- **Total objects** — total detection count
- **Processing time** — analysis time in milliseconds

### Color Coding

- 🟢 **Green box** — person detected
- 🔴 **Red box** — dangerous object detected
- 🟢 **Green panel** — safe
- 🔴 **Red panel** — threat detected

---

## Report Format

Example text report:

```
============================================================
  REPORT — SECURITY MONITORING SYSTEM
  Date: 2026-02-17 22:30:00
  File: C:\images\test.jpg
============================================================

  People detected:        3
  Dangerous objects:      1
  Total objects:          4
  Processing time:        45 ms

  *** THREAT DETECTED! ***
  Dangerous objects: knife

  Detailed list of detected objects:
------------------------------------------------------------
    1. [OK]     Person          Confidence: 92.3%
    2. [OK]     Person          Confidence: 87.1%
    3. [OK]     Person          Confidence: 74.5%
    4. [DANGER] KNIFE           Confidence: 68.2%
------------------------------------------------------------
```

---

## About YOLOv4-tiny

**YOLO (You Only Look Once)** is a neural network architecture that detects objects in a single pass, providing high-speed performance.

**YOLOv4-tiny** is a lightweight version of YOLOv4:
- Model size: ~24 MB (vs ~256 MB for full YOLOv4)
- Speed: ~30+ FPS on CPU
- Trained on **COCO dataset** (80 object classes)

It is loaded via the **cv2.dnn** (Deep Neural Network) module of OpenCV, which allows running the neural network without additional frameworks (TensorFlow, PyTorch not required).

### Performance optimization

The first forward pass through the network is always slower because OpenCV lazily allocates memory and compiles internal kernels. This application performs a **warm-up inference** on a dummy image during startup, so all subsequent detections run at full speed.

---
