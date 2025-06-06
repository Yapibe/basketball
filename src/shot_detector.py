"""Core modules for basketball shot detection and classification.

Current phase: MADE / MISSED classification
Future phases: pose analysis and Gemini feedback
"""

from __future__ import annotations
import os
import math
from pathlib import Path
from typing import Iterator, List, Tuple, Dict

import cv2
import numpy as np
import torch

# Ultralytics will lazily download weights on first run
from ultralytics import YOLO


class RimDetector:
    """Detects the hoop rim in the first N frames and caches results."""

    def __init__(self, max_frames: int = 300) -> None:
        self.max_frames = max_frames
        self._cache: Tuple[Tuple[int, int], int] | None = None

    def locate(self, frame_iter: Iterator[np.ndarray]) -> Tuple[Tuple[int, int], int]:
        if self._cache is not None:
            return self._cache

        frames = []
        for i, f in enumerate(frame_iter):
            frames.append(f)
            if i >= self.max_frames:
                break

        # Strategy 1: HoughCircles on grayscale edges
        for idx, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=100,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=120,
            )
            if circles is not None:
                x, y, r = circles[0][0]
                self._cache = ((int(x), int(y)), int(r))
                return self._cache

        # Strategy 2 TODO: YOLO hoop class
        raise RuntimeError("Rim not detected automatically. "
                           "Consider manual calibration (see README).")


class BallTracker:
    """Tracks ball centroid across frames using object detection + simple ID association."""

    def __init__(self, model_weights: str | None = None, conf: float = 0.25) -> None:
        self.model = YOLO(model_weights or "yolov8n.pt")
        self.conf = conf

    def track(self, frame_iter: Iterator[np.ndarray]) -> List[Dict]:
        tracks: List[Dict] = []
        frame_idx = 0
        for frame in frame_iter:
            results = self.model.predict(
                frame,
                conf=self.conf,
                classes=None,  # Let model decide; basketball assumed class 0/32 etc.
                verbose=False,
                device=0 if torch.cuda.is_available() else "cpu",
            )
            if results and len(results[0].boxes) > 0:
                box = results[0].boxes.xywh[0]  # take first / highest‑score box
                x, y, w, h = box.tolist()
                tracks.append({"frame": frame_idx, "x": x, "y": y})
            frame_idx += 1
        return tracks


class ShotClassifier:
    """Segments continuous tracks into shot attempts and classifies MADE / MISSED."""

    def __init__(self, rim_center: Tuple[int, int], rim_radius: int) -> None:
        self.cx, self.cy = rim_center
        self.r = rim_radius

    def _fit_parabola(self, seq: List[Dict]) -> Tuple[float, float, float]:
        # Fit y = ax^2 + bx + c in pixel space
        xs = np.array([p["x"] for p in seq])
        ys = np.array([p["y"] for p in seq])
        coeffs = np.polyfit(xs, ys, deg=2)
        return tuple(coeffs)  # a, b, c

    def classify(self, tracks: List[Dict], rim: Tuple[Tuple[int, int], int]) -> List[Dict]:
        rim_center, rim_radius = rim
        shots: List[Dict] = []
        # TODO: simple segmentation by gaps > N frames
        if not tracks:
            return shots

        # Naive single‑shot assumption:
        seq = tracks
        a, b, c = self._fit_parabola(seq)

        # solve for y = rim plane at x coordinate where ball crosses rim height (cy)
        # Simplification: use last downward point as crossing
        cross = seq[-1]
        dist = math.hypot(cross["x"] - rim_center[0], cross["y"] - rim_center[1])

        made = dist < rim_radius * 0.9  # epsilon = 10%
        shots.append(
            {
                "id": 1,
                "made": bool(made),
                "t": (seq[-1]["frame"] - seq[0]["frame"]) / 30.0,  # assuming 30 fps
            }
        )
        return shots


# ------------------------ CLI entry‑point ------------------------ #

def _frames_from_video(path: str) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Shot classification prototype")
    parser.add_argument("video_path", type=Path, help="Path to input video")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    frame_iter = list(_frames_from_video(str(args.video_path)))

    rim = RimDetector().locate(iter(frame_iter))
    print(f"Rim detected at {rim}")

    tracks = BallTracker().track(iter(frame_iter))
    print(f"Tracked {len(tracks)} ball positions")

    shots = ShotClassifier(*rim).classify(tracks, rim)
    for s in shots:
        status = "MAKE" if s["made"] else "MISS"
        print(f"Shot #{s['id']} — {status} (flight {s['t']:.2f}s)")

if __name__ == "__main__":
    main()