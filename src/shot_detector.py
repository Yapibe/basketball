import cv2
import numpy as np
from ultralytics import YOLO
from typing import Iterator, Tuple, List, Dict
import torch

class RimDetector:
    def __init__(self):
        self.rim_center = None
        self.rim_radius = None
        self.model = YOLO('yolov8n.pt')  # Will be replaced with basketball-specific model
        
    def locate(self, frame_iter: Iterator[np.ndarray]) -> Tuple[Tuple[int, int], int]:
        """Detect rim location from first 300 frames using Hough transform or YOLO."""
        frames_checked = 0
        for frame in frame_iter:
            if frames_checked >= 300:
                break
                
            # Convert to grayscale for Hough transform
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Try Hough circle detection
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                param1=50, param2=30, minRadius=20, maxRadius=100
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                # Take the first detected circle
                x, y, r = circles[0][0]
                self.rim_center = (x, y)
                self.rim_radius = r
                return self.rim_center, self.rim_radius
                
            frames_checked += 1
            
        # Fallback to YOLO if Hough fails
        # TODO: Implement YOLO-based rim detection
        return None, None

class BallTracker:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Will be replaced with basketball-specific model
        self.track_history = []
        
    def track(self, frame_iter: Iterator[np.ndarray]) -> List[Dict]:
        """Track basketball across frames using YOLO."""
        tracks = []
        frame_idx = 0
        
        for frame in frame_iter:
            results = self.model.track(frame, persist=True)
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    if box.cls == 32:  # Assuming basketball class ID
                        x, y = box.xywh[0][:2].cpu().numpy()
                        tracks.append({
                            'frame': frame_idx,
                            'x': int(x),
                            'y': int(y),
                            'confidence': float(box.conf)
                        })
            
            frame_idx += 1
            
        return tracks

class ShotClassifier:
    def __init__(self, rim_center: Tuple[int, int], rim_radius: int):
        self.rim_center = rim_center
        self.rim_radius = rim_radius
        self.rim_height = rim_center[1]  # y-coordinate is rim height
        
    def _fit_parabola(self, points: List[Dict]) -> Tuple[float, float, float]:
        """Fit parabola to ball trajectory using RANSAC."""
        x = np.array([p['x'] for p in points])
        y = np.array([p['y'] for p in points])
        
        # Simple least squares fit for now
        A = np.vstack([x**2, x, np.ones(len(x))]).T
        a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return a, b, c
        
    def _check_made(self, trajectory: List[Dict]) -> Tuple[bool, float]:
        """Determine if shot was made based on trajectory analysis."""
        if len(trajectory) < 15:
            return False, 0.0
            
        # Get recent points
        recent = trajectory[-15:]
        a, b, c = self._fit_parabola(recent)
        
        # Find where parabola crosses rim height
        x = np.array([p['x'] for p in recent])
        y = np.array([p['y'] for p in recent])
        
        # Check if ball is descending
        descending = np.all(np.diff(y) < 0)
        
        # Calculate distance from rim center at crossing point
        x_cross = (-b + np.sqrt(b**2 - 4*a*(c - self.rim_height))) / (2*a)
        y_cross = self.rim_height
        
        dist = np.sqrt((x_cross - self.rim_center[0])**2 + 
                      (y_cross - self.rim_center[1])**2)
        
        # Shot is made if ball passes through rim and continues descending
        made = (dist < self.rim_radius - 5) and descending
        
        # Calculate flight time
        flight_time = (recent[-1]['frame'] - recent[0]['frame']) / 30.0  # Assuming 30 fps
        
        return made, flight_time
        
    def classify(self, tracks: List[Dict]) -> List[Dict]:
        """Classify shots based on ball tracking data."""
        shots = []
        current_trajectory = []
        
        for track in tracks:
            current_trajectory.append(track)
            
            # If we have enough points and ball is descending
            if len(current_trajectory) >= 15:
                made, flight_time = self._check_made(current_trajectory)
                
                if made or flight_time > 1.0:  # Shot completed or too long
                    shots.append({
                        'id': len(shots) + 1,
                        'made': made,
                        't': flight_time
                    })
                    current_trajectory = []
                    
        return shots

def main(video_path: str):
    """Command-line entry point for shot analysis."""
    cap = cv2.VideoCapture(video_path)
    
    def frame_generator():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    
    # Detect rim
    rim_detector = RimDetector()
    rim_center, rim_radius = rim_detector.locate(frame_generator())
    
    if rim_center is None:
        print("Error: Could not detect rim")
        return
        
    # Track ball
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video
    ball_tracker = BallTracker()
    tracks = ball_tracker.track(frame_generator())
    
    # Classify shots
    classifier = ShotClassifier(rim_center, rim_radius)
    shots = classifier.classify(tracks)
    
    # Print results
    for shot in shots:
        print(f"Shot #{shot['id']} — {'MADE' if shot['made'] else 'MISS'} (flight {shot['t']:.2f}s)")
    
    cap.release()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m shot_detector <video_path>")
        sys.exit(1)
    main(sys.argv[1]) 