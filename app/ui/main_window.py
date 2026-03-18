"""
app/ui/main_window.py — Security Monitoring GUI (CustomTkinter).

Layout:
  [ Sidebar (controls) ] | [ Canvas (video/image feed) ] | [ Stats panel ]
"""

import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk

import customtkinter as ctk
from tkinter import filedialog, messagebox

from app.core.detector import ObjectDetector, DetectionResult
from app.config import YOLO_POSE_MODEL, CAMERA_INDEX, CAMERA_INDICES, CAMERA_PREFER_USB, MAX_CAMERAS

# ──────────────────────────────────────────────────────────────
# Appearance
# ──────────────────────────────────────────────────────────────

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

ACCENT_GREEN = "#22c55e"
ACCENT_RED   = "#ef4444"
ACCENT_BLUE  = "#3b82f6"
BG_CARD      = "#1e1e26"
TEXT_SUBTLE  = "#94a3b8"


# ──────────────────────────────────────────────────────────────
# Main Application Window
# ──────────────────────────────────────────────────────────────

class MainWindow(ctk.CTk):
    """Main application window — Security Intelligence System with Multi-Camera Support."""

    def __init__(self):
        super().__init__()

        self.title("SECURITY INTELLIGENCE SYSTEM (MVP)")
        self.geometry("1400x900")
        self.minsize(1100, 750)

        # ── State ──────────────────────────────────────────────
        self.detector: ObjectDetector | None = None
        self.current_image  = None
        self.annotated_image = None
        self.current_result: DetectionResult | None = None
        self.image_path = ""

        self.camera_active = False
        self.camera_threads = []
        self.camera_caps = []
        self.canvases = [] # Grid of canvases
        self.last_results = {} # per camera

        self.video_active = False
        self.video_thread = None
        self.video_path   = ""

        # ── Tkinter variables ──────────────────────────────────
        self.status_var    = ctk.StringVar(value="Initializing AI Core...")
        self.confidence_var = ctk.IntVar(value=30) # Default for pose

        # ── Build UI ───────────────────────────────────────────
        self._setup_grid()
        self._build_sidebar()
        self._build_main_view()
        self._build_stats_panel()

        # Load AI model in background thread
        threading.Thread(target=self._init_detector, daemon=True).start()

    # ══════════════════════════════════════════════════════════
    # Layout helpers
    # ══════════════════════════════════════════════════════════

    def _setup_grid(self):
        self.grid_columnconfigure(1, weight=1)   # main view stretches
        self.grid_columnconfigure(2, weight=0)   # stats panel fixed
        self.grid_rowconfigure(0, weight=1)

    # ══════════════════════════════════════════════════════════
    # UI — Sidebar
    # ══════════════════════════════════════════════════════════

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            self.sidebar, text="AI SECURITY",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(30, 10))

        ctk.CTkLabel(
            self.sidebar, text="Skeleton Detection v3.0 (MVP)",
            font=ctk.CTkFont(size=12), text_color=TEXT_SUBTLE,
        ).grid(row=1, column=0, padx=20, pady=(0, 30))

        btn = {"padx": 20, "pady": 10, "sticky": "ew"}

        self.btn_load = ctk.CTkButton(
            self.sidebar, text="📁 LOAD IMAGE",
            command=self._load_image, height=45,
            font=ctk.CTkFont(weight="bold"),
        )
        self.btn_load.grid(row=2, column=0, **btn)

        self.btn_video = ctk.CTkButton(
            self.sidebar, text="🎬 LOAD VIDEO",
            command=self._load_video, height=45,
            fg_color="#7c3aed", hover_color="#6d28d9",
            font=ctk.CTkFont(weight="bold"),
        )
        self.btn_video.grid(row=3, column=0, **btn)

        self.btn_camera = ctk.CTkButton(
            self.sidebar, text="📹 START CAMERAS",
            command=self._toggle_cameras, height=45,
            fg_color=ACCENT_GREEN, hover_color="#16a34a",
            font=ctk.CTkFont(weight="bold"),
        )
        self.btn_camera.grid(row=4, column=0, **btn)

        self.btn_save = ctk.CTkButton(
            self.sidebar, text="💾 SAVE RESULT",
            command=self._save_result, height=45,
            fg_color="transparent", border_width=2,
            font=ctk.CTkFont(weight="bold"),
        )
        self.btn_save.grid(row=5, column=0, **btn)

        # Confidence slider
        ctk.CTkLabel(
            self.sidebar, text="Confidence Threshold",
            font=ctk.CTkFont(size=13), text_color=TEXT_SUBTLE,
        ).grid(row=8, column=0, padx=20, pady=(30, 0), sticky="w")

        self.conf_slider = ctk.CTkSlider(
            self.sidebar, from_=10, to=90,
            variable=self.confidence_var,
            command=self._on_confidence_change,
        )
        self.conf_slider.grid(row=9, column=0, padx=20, pady=(5, 5), sticky="ew")

        self.conf_val_label = ctk.CTkLabel(
            self.sidebar, text=f"{self.confidence_var.get()}%",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.conf_val_label.grid(row=10, column=0, padx=20, pady=0, sticky="e")

        # Status badge
        self.status_container = ctk.CTkFrame(
            self.sidebar, corner_radius=10, fg_color="#111111",
        )
        self.status_container.grid(row=11, column=0, padx=20, pady=20, sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.status_container, textvariable=self.status_var,
            font=ctk.CTkFont(size=11), text_color=ACCENT_BLUE,
        )
        self.status_label.pack(pady=10)

    # ══════════════════════════════════════════════════════════
    # UI — Main grid view
    # ══════════════════════════════════════════════════════════

    def _build_main_view(self):
        self.main_container = ctk.CTkFrame(self, fg_color="#0a0a0a", corner_radius=0)
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=2, pady=0)

        ctk.CTkLabel(
            self.main_container, text="MULTI-CAMERA SURVEILLANCE GRID",
            font=ctk.CTkFont(size=12, weight="bold", slant="italic"),
            text_color="#444444",
        ).pack(pady=10, anchor="n")

        self.grid_frame = ctk.CTkFrame(self.main_container, fg_color="#0a0a0a")
        self.grid_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # We'll dynamically populate grid_frame with canvases when cameras start
        self._setup_canvases(len(CAMERA_INDICES))

        self.placeholder = ctk.CTkLabel(
            self.main_container,
            text="WAITING FOR INPUT\n\n[ Load Image, Video or Start Cameras ]",
            font=ctk.CTkFont(size=18, slant="italic"),
            text_color="#222222",
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")

        self.progress_bar = ctk.CTkProgressBar(self.main_container, width=300)
        self.progress_bar.place(relx=0.5, rely=0.6, anchor="center")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()

    def _setup_canvases(self, count):
        """Create a grid of canvases for multiple cameras."""
        # Clear existing
        for c in self.canvases:
            c.destroy()
        self.canvases = []

        if count == 0: return
        
        # Determine grid size (MVP: 2x2 max)
        cols = 2 if count > 1 else 1
        rows = (count + 1) // 2

        for i in range(count):
            c = ctk.CTkCanvas(self.grid_frame, bg="#0f0f0f", highlightthickness=1, highlightbackground="#1e1e26")
            c.grid(row=i // cols, column=i % cols, sticky="nsew", padx=2, pady=2)
            self.grid_frame.grid_columnconfigure(i % cols, weight=1)
            self.grid_frame.grid_rowconfigure(i // cols, weight=1)
            self.canvases.append(c)
            c.bind("<Configure>", lambda e, idx=i: self._on_canvas_resize(idx))

    # ══════════════════════════════════════════════════════════
    # AI Core — initialisation
    # ══════════════════════════════════════════════════════════

    def _init_detector(self):
        try:
            # ObjectDetector now handles its own models internally
            self.detector = ObjectDetector()
            self.detector.confidence_threshold = self.confidence_var.get() / 100.0
            self.after(0, self._on_detector_ready)
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self._on_detector_error(err_msg))

    def _on_detector_ready(self):
        self.status_var.set("AI Core: Pose Model Ready")
        self.progress_bar.stop()
        self.progress_bar.place_forget()

    def _on_detector_error(self, msg: str):
        self.status_var.set(f"Critical Error: {msg}")
        self.progress_bar.stop()
        messagebox.showerror("AI Init Error", f"Model failure:\n{msg}")

    def _on_confidence_change(self, value):
        self.conf_val_label.configure(text=f"{int(value)}%")
        if self.detector:
            self.detector.confidence_threshold = int(value) / 100.0

    # ══════════════════════════════════════════════════════════
    # Camera Management
    # ══════════════════════════════════════════════════════════

    def _toggle_cameras(self):
        if self.video_active:
            self._stop_video()

        if self.camera_active:
            self.camera_active = False
            self.btn_camera.configure(text="📹 START CAMERAS", fg_color=ACCENT_GREEN)
            self.placeholder.place(relx=0.5, rely=0.5, anchor="center")
            # Stop all captures
            for cap in self.camera_caps:
                cap.release()
            self.camera_caps = []
            return

        self.camera_active = True
        self.btn_camera.configure(text="🛑 STOP CAMERAS", fg_color=ACCENT_RED)
        self.placeholder.place_forget()
        
        # --- Auto-discovery ---
        active_indices = self._discover_cameras()
        if not active_indices:
            self.camera_active = False
            self.btn_camera.configure(text="📹 START CAMERAS", fg_color=ACCENT_GREEN)
            self.placeholder.place(relx=0.5, rely=0.5, anchor="center")
            messagebox.showwarning("Camera Error", "No cameras detected. Please check connection.")
            return

        self.status_var.set(f"Monitoring {len(active_indices)} camera(s)...")

        # Setup canvases again based on found cameras
        self._setup_canvases(len(active_indices))
        self.current_camera_indices = active_indices # Store locally for workers

        for idx in active_indices:
            t = threading.Thread(target=self._camera_worker, args=(idx,), daemon=True)
            t.start()
            self.camera_threads.append(t)

    def _discover_cameras(self) -> list:
        """Thoroughly find active camera indices and log findings to console."""
        found = []
        
        # We will check indices 0 to 4
        print("[DEBUG] Scanning for cameras...")
        self.status_var.set("Scanning for cameras...")
        
        for idx in range(5):
            # Try multiple backends if default fails
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) # DSHOW is usually better for Windows USB
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx) # Fallback to default
                
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"[DEBUG] Found working camera at index {idx}")
                    found.append(idx)
                cap.release()
        
        print(f"[DEBUG] Total cameras found: {len(found)} (Indices: {found})")
        
        # If we have a specific preference in config but it wasn't 'found', 
        # let's try to return what we found but prioritize USB (usually indices 1-4)
        if len(found) > 1:
            # Sort so that index 1, 2, 3... come before 0 (builtin)
            found.sort(key=lambda x: 1 if x == 0 else 0)
            
        return found

    def _camera_worker(self, cam_idx):
        """Worker for each camera stream."""
        cap = cv2.VideoCapture(cam_idx)  # default backend
        if not cap.isOpened():
            self.after(0, lambda: self.status_var.set(f"Err: Cam {cam_idx} not found"))
            return

        self.camera_caps.append(cap)
        # Optimized resolution for MVP performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            canvas_idx = self.current_camera_indices.index(cam_idx)
        except (ValueError, AttributeError):
            cap.release()
            return
        
        while self.camera_active:
            ret, frame = cap.read()
            if not ret: break

            if self.detector:
                # OPTIMIZATION: Process detection
                result = self.detector.detect(frame)
                annotated = self.detector.draw_detections(frame, result)
                self.last_results[canvas_idx] = (annotated, result)
                
                # Update UI (only if active)
                if self.camera_active:
                    self.after(0, self._update_camera_frame, canvas_idx, annotated, result)
            else:
                self.after(0, self._display_frame, canvas_idx, frame)
            
            # Control FPS (approx 30fps)
            time.sleep(0.01)

        cap.release()

    def _update_camera_frame(self, canvas_idx, image, result):
        if not self.camera_active: return
        self._display_frame(canvas_idx, image)
        
        # Aggregate stats from all cameras for the panel
        total_people = sum(r[1].person_count for r in self.last_results.values() if r)
        max_latency = max(r[1].processing_time_ms for r in self.last_results.values() if r)
        
        self.card_people.configure(text=str(total_people))
        self.card_perf.configure(text=f"{max_latency:.0f}")

    def _display_frame(self, canvas_idx, image):
        if canvas_idx >= len(self.canvases): return
        canvas = self.canvases[canvas_idx]
        
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if cw < 10 or ch < 10: return

        ih, iw = image.shape[:2]
        ratio = min(cw / iw, ch / ih, 1.0)
        nw, nh = int(iw * ratio), int(ih * ratio)

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        canvas._current_photo = photo 

    def _on_canvas_resize(self, idx):
        if idx in self.last_results:
            self._display_frame(idx, self.last_results[idx][0])

    # ══════════════════════════════════════════════════════════
    # Rest of the methods (Image/Video) adapted to single canvas view
    # ══════════════════════════════════════════════════════════

    def _load_image(self):
        # Implementation for single image - use canvas 0 or reset grid
        if self.camera_active: self._toggle_cameras()
        if self.video_active: self._stop_video()
        
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path: return
        
        img_data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if image is not None:
            self._setup_canvases(1) # Reset to 1 for image
            self.placeholder.place_forget()
            if self.detector:
                res = self.detector.detect(image)
                ann = self.detector.draw_detections(image, res)
                self.last_results[0] = (ann, res)
                self._display_frame(0, ann)
                self._update_camera_frame(0, ann, res)
            else:
                self._display_frame(0, image)

    def _load_video(self):
        if self.camera_active: self._toggle_cameras()
        if self.video_active: self._stop_video()
        
        path = filedialog.askopenfilename()
        if not path: return
        
        self.video_active = True
        self._setup_canvases(1)
        self.placeholder.place_forget()
        self.btn_video.configure(text="⏹ STOP VIDEO", fg_color=ACCENT_RED, command=self._stop_video)
        
        self.video_thread = threading.Thread(target=self._video_loop, args=(path,), daemon=True)
        self.video_thread.start()

    def _stop_video(self):
        self.video_active = False
        self.btn_video.configure(text="🎬 LOAD VIDEO", fg_color="#7c3aed", command=self._load_video)

    def _video_loop(self, path):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / fps
        
        while self.video_active:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret: break
            
            if self.detector:
                res = self.detector.detect(frame)
                ann = self.detector.draw_detections(frame, res)
                self.last_results[0] = (ann, res)
                self.after(0, self._update_camera_frame, 0, ann, res)
            else:
                self.after(0, self._display_frame, 0, frame)
            
            st = delay - (time.time() - t0)
            if st > 0: time.sleep(st)
        
        cap.release()
        self.after(0, self._stop_video)

    def _save_result(self):
        if 0 in self.last_results:
            ann = self.last_results[0][0]
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                cv2.imwrite(path, ann)
                messagebox.showinfo("Saved", "Saved successfully")

    # ══════════════════════════════════════════════════════════
    # Stats & Stats helper
    # ══════════════════════════════════════════════════════════

    def _build_stats_panel(self):
        self.stats_panel = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.stats_panel.grid(row=0, column=2, sticky="nsew")

        ctk.CTkLabel(
            self.stats_panel, text="LIVE STATISTICS",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(pady=(30, 20), padx=20, anchor="w")

        self.card_people = self._create_card(self.stats_panel, "People Total", "0")
        self.card_perf   = self._create_card(self.stats_panel, "Max Latency (ms)", "—")

        self.alert_frame = ctk.CTkFrame(self.stats_panel, height=80, fg_color="#18181b")
        self.alert_frame.pack(fill="x", padx=20, pady=20)
        self.alert_frame.pack_propagate(False)

        self.alert_label = ctk.CTkLabel(
            self.alert_frame, text="SYSTEM READY",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=ACCENT_GREEN,
        )
        self.alert_label.pack(expand=True)

    def _create_card(self, parent, title: str, value: str, color=None):
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, height=100)
        card.pack(fill="x", padx=20, pady=8)
        card.pack_propagate(False)

        ctk.CTkLabel(
            card, text=title.upper(),
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_SUBTLE,
        ).pack(anchor="w", padx=15, pady=(15, 0))

        val_label = ctk.CTkLabel(
            card, text=value,
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=color if color else "white",
        )
        val_label.pack(anchor="w", padx=15, pady=(0, 10))
        return val_label

    def run(self):
        self.mainloop()
