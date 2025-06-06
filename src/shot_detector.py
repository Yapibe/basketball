"""
Core classes for basketball shot analysis.
Handles rim detection, ball tracking, and shot classification.
"""
import cv2
import numpy as np
from typing import Iterator, Tuple, List, Dict
from ultralytics import YOLO
import torch

class RimDetector:
    def __init__(self):
        self._cache = None  # (center, radius)
        self._model = YOLO('yolov8n.pt')  # Fallback to YOLO if Hough fails
        
    def locate(self, frame_iter: Iterator[np.ndarray]) -> Tuple[Tuple[int, int], int]:
        """Detect rim in first 300 frames using Hough transform or YOLO."""
        if self._cache:
            return self._cache
            
        frames_checked = 0
        for frame in frame_iter:
            if frames_checked >= 300:
                break
                
            # Try Hough transform first
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                param1=50, param2=30, minRadius=20, maxRadius=100
            )
            
            if circles is not None:
                # Take the largest circle
                circles = np.uint16(np.around(circles))
                largest = max(circles[0], key=lambda x: x[2])
                self._cache = ((largest[0], largest[1]), largest[2])
                return self._cache
                
            frames_checked += 1
            
        # Fallback to YOLO if Hough fails
        # TODO: Implement YOLO rim detection
        raise NotImplementedError("Rim detection failed. TODO: Add YOLO fallback")

class BallTracker:
    def __init__(self):
        self._model = YOLO('yolov8n.pt')  # Will be replaced with basketball-specific weights
        self._tracker = None  # TODO: Add SORT tracker for smoother tracking
        
    def track(self, frame_iter: Iterator[np.ndarray]) -> List[Dict]:
        """Track ball across frames, return list of detections."""
        tracks = []
        frame_idx = 0
        
        for frame in frame_iter:
            # Run YOLO detection
            results = self._model(frame, classes=[32])  # COCO class 32 is sports ball
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]  # Take highest confidence detection
                x, y = int(box.xywh[0][0]), int(box.xywh[0][1])
                tracks.append({
                    'frame': frame_idx,
                    'x': x,
                    'y': y,
                    'conf': float(box.conf[0])
                })
            frame_idx += 1
            
        return tracks

class ShotClassifier:
    def __init__(self):
        self._rim_height = None  # Will be set from rim detection
        
    def _fit_parabola(self, points: List[Dict], window: int = 15) -> Tuple[float, float, float]:
        """Fit parabola to recent ball positions using RANSAC."""
        if len(points) < window:
            return None
            
        recent = points[-window:]
        x = np.array([p['x'] for p in recent])
        y = np.array([p['y'] for p in recent])
        
        # Simple least squares fit
        A = np.vstack([x**2, x, np.ones(len(x))]).T
        a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return a, b, c
        
    def classify(self, tracks: List[Dict], rim: Tuple[Tuple[int, int], int]) -> List[Dict]:
        """Classify each shot attempt as MADE or MISSED."""
        if not tracks:
            return []
            
        shots = []
        rim_center, rim_radius = rim
        rim_x, rim_y = rim_center
        
        # Group tracks into potential shots
        current_shot = []
        for track in tracks:
            if not current_shot or track['frame'] - current_shot[-1]['frame'] <= 2:
                current_shot.append(track)
            else:
                if len(current_shot) >= 10:  # Minimum frames for a shot
                    # Fit parabola and check if ball went through rim
                    parabola = self._fit_parabola(current_shot)
                    if parabola:
                        a, b, c = parabola
                        # Find where parabola crosses rim height
                        # TODO: Implement proper intersection check
                        made = False  # Placeholder
                        shots.append({
                            'id': len(shots) + 1,
                            'made': made,
                            't': (current_shot[-1]['frame'] - current_shot[0]['frame']) / 30.0
                        })
                current_shot = [track]
                
        return shots

def main(video_path: str, cpu_only: bool = False):
    """CLI entry point for headless analysis."""
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    def frame_iter():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
            
    detector = RimDetector()
    tracker = BallTracker()
    classifier = ShotClassifier()
    
    try:
        rim = detector.locate(frame_iter())
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        tracks = tracker.track(frame_iter())
        shots = classifier.classify(tracks, rim)
        
        for shot in shots:
            print(f"Shot #{shot['id']} — {'MADE' if shot['made'] else 'MISS'} (flight {shot['t']:.2f}s)")
            
    finally:
        cap.release()

if __name__ == "__main__":
    import sys
    import os
    main(sys.argv[1], "--cpu-only" in sys.argv) 