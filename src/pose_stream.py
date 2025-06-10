import cv2
import mediapipe as mp
import torch
import time
import logging
import sys
import platform
import os
from typing import Tuple, Generator, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PoseDetector")

def get_system_info():
    """Get detailed system information for debugging."""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cv2_version": cv2.__version__,
        "mediapipe_version": mp.__version__,
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = torch.cuda.get_device_capability(0)
    
    return info

def log_system_info():
    """Log detailed system information."""
    info = get_system_info()
    logger.info("==== System Information ====")
    for key, value in info.items():
        logger.info(f"{key}: {value}")
    logger.info("============================")

def select_device(force_cpu: bool = False) -> str:
    """Detect and return the best available device for inference."""
    logger.info("Selecting device for inference...")
    
    if force_cpu:
        logger.info("Forcing CPU usage as requested")
        return "CPU"
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            cuda_capability = torch.cuda.get_device_capability(i)
            logger.info(f"GPU {i}: {device_name} (CUDA {cuda_capability[0]}.{cuda_capability[1]})")
        
        # Check CUDA environment variables
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        
        # Try to initialize CUDA context
        try:
            # Try to create a tensor on GPU to see if it works
            test_tensor = torch.zeros(1).cuda()
            logger.info("Successfully created test tensor on GPU")
            
            # Set PyTorch to use CUDA
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            logger.info("Set default tensor type to CUDA")
            
            return "GPU"
        except Exception as e:
            logger.error(f"Error initializing CUDA: {str(e)}")
            logger.info("Falling back to CPU")
            return "CPU"
    
    logger.info("No GPU detected, using CPU")
    return "CPU"

class PoseDetector:
    def __init__(self, use_gpu: bool = True, min_detection_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Log device usage
        device_str = "GPU" if use_gpu else "CPU"
        logger.info(f"Initializing MediaPipe Pose on {device_str}")
        logger.info(f"Detection confidence: {min_detection_confidence}")
        
        # Initialize MediaPipe Pose with GPU if available
        try:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,  # 0=Lite, 1=Full, 2=Heavy
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Pose initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MediaPipe Pose: {str(e)}")
            raise
        
        # FPS calculation
        self.prev_time = time.time()
        self.fps = 0
        self.use_gpu = use_gpu
        self.frame_count = 0
        
    def process_frame(self, frame: cv2.Mat) -> Tuple[cv2.Mat, float]:
        """Process a single frame and return annotated frame with FPS."""
        self.frame_count += 1
        
        # Log every 100 frames
        if self.frame_count % 100 == 0:
            logger.info(f"Processing frame {self.frame_count}, current FPS: {self.fps:.1f}")
        
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
        
        # Add GPU/CPU indicator
        device_text = "GPU ⚡" if self.use_gpu else "CPU"
        cv2.putText(frame, device_text, (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, self.fps
    
    def __del__(self):
        """Clean up resources."""
        logger.info("Cleaning up PoseDetector resources")
        if hasattr(self, 'pose'):
            self.pose.close()

def run_pose(cam_index=0, use_gpu=True, debug=False):
    """
    Yields (annotated_frame, fps) tuples in BGR format.
    """
    logger.info(f"Starting pose detection with camera {cam_index}, GPU={use_gpu}, debug={debug}")
    
    # Initialize camera with DirectShow on Windows
    try:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.error(f"Failed to open camera {cam_index}")
            raise RuntimeError(f"Failed to open camera {cam_index}")
            
        logger.info("Camera opened successfully")
        
        # Get camera properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera properties: {width}x{height} @ {original_fps} FPS")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info("Camera properties set to 1280x720")
    except Exception as e:
        logger.error(f"Error initializing camera: {str(e)}")
        raise
    
    # Initialize pose detector
    try:
        detector = PoseDetector(use_gpu=use_gpu)
        logger.info("Pose detector initialized")
    except Exception as e:
        logger.error(f"Error initializing pose detector: {str(e)}")
        if cap.isOpened():
            cap.release()
        raise
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break
                
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Read {frame_count} frames from camera")
                
            # Process frame
            try:
                annotated_frame, fps = detector.process_frame(frame)
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                continue
            
            # Show debug window if requested
            if debug:
                cv2.imshow("Pose Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Quit requested from debug window")
                    break
            
            yield annotated_frame, fps
            
    except Exception as e:
        logger.error(f"Error in run_pose: {str(e)}")
    finally:
        logger.info("Cleaning up resources")
        cap.release()
        if debug:
            cv2.destroyAllWindows()

# Log system info on module import
log_system_info() 