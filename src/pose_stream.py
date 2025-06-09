import cv2
import mediapipe as mp
import torch
import time
from typing import Tuple, Generator, Optional

def select_device(force_cpu: bool = False) -> str:
    """Detect and return the best available device for inference."""
    if force_cpu:
        return "CPU"
    if torch.cuda.is_available():
        return "GPU"
    return "CPU"

class PoseDetector:
    def __init__(self, use_gpu: bool = True):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Pose with GPU if available
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0=Lite, 1=Full, 2=Heavy
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # FPS calculation
        self.prev_time = time.time()
        self.fps = 0
        
    def process_frame(self, frame: cv2.Mat) -> Tuple[cv2.Mat, float]:
        """Process a single frame and return annotated frame with FPS."""
        # Resize frame for better performance
        frame = cv2.resize(frame, (960, 540))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Calculate FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
        # Add FPS text
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, self.fps

def run_pose(cam_index: int = 0, use_gpu: bool = True, debug: bool = False) -> Generator[Tuple[cv2.Mat, float], None, None]:
    """
    Main pose detection loop that yields (annotated_frame, fps) tuples.
    
    Args:
        cam_index: Camera device index
        use_gpu: Whether to use GPU if available
        debug: Whether to show OpenCV window
    
    Yields:
        Tuple of (annotated_frame, fps)
    """
    # Initialize camera with DirectShow on Windows
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {cam_index}")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize pose detector
    detector = PoseDetector(use_gpu=use_gpu)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            annotated_frame, fps = detector.process_frame(frame)
            
            # Show debug window if requested
            if debug:
                cv2.imshow("Pose Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            yield annotated_frame, fps
            
    finally:
        cap.release()
        if debug:
            cv2.destroyAllWindows() 