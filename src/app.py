import gradio as gr
import cv2
import numpy as np
from pose_stream import PoseDetector, select_device, log_system_info
import argparse
import time
import psutil
import torch
import logging
import sys
import os
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PoseApp")

def check_gpu_status():
    """Check and log detailed GPU status information."""
    logger.info("==== GPU Status Check ====")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version
        cuda_version = torch.version.cuda
        logger.info(f"CUDA version: {cuda_version}")
        
        # Get device count
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA device count: {device_count}")
        
        # Get device properties for each device
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            logger.info(f"GPU {i}: {device_name} (CUDA capability {device_capability[0]}.{device_capability[1]})")
            
        # Check current device
        current_device = torch.cuda.current_device()
        logger.info(f"Current CUDA device: {current_device}")
        
        # Check memory usage
        try:
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            memory_cached = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
            logger.info(f"CUDA memory allocated: {memory_allocated:.2f} MB")
            logger.info(f"CUDA memory reserved: {memory_cached:.2f} MB")
        except Exception as e:
            logger.error(f"Error getting CUDA memory info: {str(e)}")
            
        # Check environment variables
        for env_var in ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER']:
            value = os.environ.get(env_var, 'Not set')
            logger.info(f"{env_var}: {value}")
    
    # Test CUDA functionality
    if cuda_available:
        try:
            logger.info("Attempting to create a CUDA tensor...")
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            logger.info(f"Successfully created tensor on CUDA: {x}")
            logger.info("GPU is working correctly")
        except Exception as e:
            logger.error(f"Failed to create CUDA tensor: {str(e)}")
            logger.info("GPU appears to be available but not functioning correctly")
    
    logger.info("==========================")
    return cuda_available

def create_interface(debug: bool = False):
    """Create and launch the Gradio interface."""
    # Check GPU status at startup
    logger.info("Starting Pose Detection app")
    gpu_available = check_gpu_status()
    
    with gr.Blocks(title="Live Pose Detection") as demo:
        gr.Markdown("# 🏃 Live Pose Detection")
        
        # Initialize state
        pose_detector = None
        device = None
        
        # Check GPU availability at startup
        has_gpu = torch.cuda.is_available()
        gpu_info = "No GPU detected"
        if has_gpu:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_info = f"{gpu_name} (CUDA {torch.version.cuda})"
                logger.info(f"Found GPU: {gpu_info}")
            except Exception as e:
                logger.error(f"Error getting GPU info: {str(e)}")
                gpu_info = "Error detecting GPU details"
        
        with gr.Row():
            with gr.Column():
                # Controls
                with gr.Row():
                    cpu_only = gr.Checkbox(
                        label="Force CPU Mode",
                        value=False,
                        info=f"GPU Available: {has_gpu} ({gpu_info})"
                    )
                
                # Performance controls
                with gr.Row():
                    fps_target = gr.Slider(
                        label="Target FPS",
                        minimum=15,
                        maximum=60,
                        value=30,
                        step=1,
                        info="Target frames per second"
                    )
                    confidence = gr.Slider(
                        label="Detection Confidence",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        info="Minimum confidence for pose detection"
                    )
                
                # Status indicators
                with gr.Row():
                    fps_display = gr.Textbox(
                        label="FPS",
                        value="0.0",
                        interactive=False
                    )
                    device_status_box = gr.Textbox(
                        label="Device",
                        value=f"Available: {'GPU' if has_gpu else 'CPU only'}",
                        interactive=False
                    )
                
                # Buttons
                with gr.Row():
                    apply_button = gr.Button("Apply Settings", variant="primary")
                    restart_button = gr.Button("Restart Stream", variant="secondary")
                    debug_button = gr.Button("Debug GPU", variant="secondary")
                
                # Debug panel (collapsible)
                with gr.Accordion("Debug Info", open=False):
                    with gr.Row():
                        capture_time = gr.Textbox(
                            label="Capture Time (ms)",
                            value="0.0",
                            interactive=False
                        )
                        process_time = gr.Textbox(
                            label="Process Time (ms)",
                            value="0.0",
                            interactive=False
                        )
                        memory_usage = gr.Textbox(
                            label="Memory Usage (MB)",
                            value="0.0",
                            interactive=False
                        )
                    
                    gpu_debug_info = gr.Textbox(
                        label="GPU Debug Info",
                        value=f"GPU Available: {has_gpu}\nGPU Name: {gpu_info}",
                        lines=10,
                        interactive=False
                    )
            
            # Camera input and output
            with gr.Column():
                camera = gr.Image(
                    label="Pose Detection",
                    height=540,
                    width=960,
                    sources=["webcam"],
                    type="numpy",
                    streaming=True
                )
        
        def update_debug_info(capture_ms: float, process_ms: float):
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            return f"{capture_ms:.1f}", f"{process_ms:.1f}", f"{memory:.1f}"
        
        def process_frame(frame: np.ndarray, cpu_only: bool, confidence: float, fps_value: int) -> Tuple[np.ndarray, str, str, str, str, str]:
            nonlocal pose_detector, device
            
            try:
                # Initialize device and detector if not already done
                if device is None or pose_detector is None:
                    logger.info(f"Initializing detector with CPU only: {cpu_only}")
                    device = select_device(force_cpu=cpu_only)
                    logger.info(f"Selected device: {device}")
                    pose_detector = PoseDetector(
                        use_gpu=(device == "GPU"),
                        min_detection_confidence=confidence
                    )
                
                # Timing for processing
                process_start = time.time()
                frame, fps = pose_detector.process_frame(frame)
                process_time = (time.time() - process_start) * 1000
                
                # Add device indicator to frame
                device_text = f"Device: {device} | Target FPS: {fps_value}"
                cv2.putText(frame, device_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update debug info
                debug_info = update_debug_info(0.0, process_time)  # Capture time is handled by Gradio
                
                return (
                    frame,
                    f"{fps:.1f}",
                    f"{device} {'⚡' if device == 'GPU' else ''}",
                    *debug_info
                )
                
            except Exception as e:
                logger.error(f"Error in process_frame: {str(e)}")
                pose_detector = None
                device = None
                return (
                    None,
                    "Error",
                    f"Error: {str(e)}",
                    "0.0",
                    "0.0",
                    "0.0"
                )
        
        def update_settings(fps_value: int):
            logger.info(f"Updating settings: FPS={fps_value}")
            return f"Target: {fps_value}", "Settings updated. Use Restart to apply."
        
        def reset_detector():
            nonlocal pose_detector, device
            logger.info("Resetting detector")
            pose_detector = None
            device = None
            return "Detector reset. Ready for new settings."
        
        def debug_gpu_info():
            logger.info("Running GPU debug diagnostics")
            gpu_available = check_gpu_status()
            
            debug_text = [f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
            debug_text.append(f"PyTorch: {torch.__version__}")
            debug_text.append(f"CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                try:
                    debug_text.append(f"CUDA version: {torch.version.cuda}")
                    debug_text.append(f"Device count: {torch.cuda.device_count()}")
                    debug_text.append(f"Current device: {torch.cuda.current_device()}")
                    for i in range(torch.cuda.device_count()):
                        name = torch.cuda.get_device_name(i)
                        capability = torch.cuda.get_device_capability(i)
                        debug_text.append(f"GPU {i}: {name} (capability: {capability[0]}.{capability[1]})")
                    
                    mem_allocated = torch.cuda.memory_allocated() / (1024**2)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
                    debug_text.append(f"Memory allocated: {mem_allocated:.2f} MB")
                    debug_text.append(f"Memory reserved: {mem_reserved:.2f} MB")
                    
                    # Test tensor creation
                    try:
                        x = torch.ones(1).cuda()
                        debug_text.append("✅ Successfully created CUDA tensor")
                    except Exception as e:
                        debug_text.append(f"❌ Failed to create CUDA tensor: {str(e)}")
                except Exception as e:
                    debug_text.append(f"Error querying GPU: {str(e)}")
            else:
                debug_text.append("No CUDA device available")
            
            return "\n".join(debug_text)
        
        # Set up the video processing
        camera.stream(
            fn=process_frame,
            inputs=[camera, cpu_only, confidence, fps_target],
            outputs=[camera, fps_display, device_status_box, capture_time, process_time, memory_usage],
            stream_every=0.1  # 10 updates per second to avoid overloading
        )
        
        # Apply settings button
        apply_button.click(
            fn=update_settings,
            inputs=[fps_target],
            outputs=[fps_display, device_status_box]
        )
        
        # Restart button
        restart_button.click(
            fn=reset_detector,
            outputs=[device_status_box]
        )
        
        # Debug GPU button
        debug_button.click(
            fn=debug_gpu_info,
            outputs=[gpu_debug_info]
        )
        
        # FPS slider change handler
        fps_target.change(
            fn=lambda fps: f"Target: {fps} (click Apply to update)",
            inputs=[fps_target],
            outputs=[fps_display]
        )
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Pose Detection")
    parser.add_argument("--debug", action="store_true", help="Show OpenCV debug window")
    args = parser.parse_args()
    
    # Log system info
    log_system_info()
    
    demo = create_interface(debug=args.debug)
    demo.launch(share=False) 