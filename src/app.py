import gradio as gr
import cv2
import numpy as np
from pose_stream import run_pose, select_device
import argparse

def process_frame(frame, fps):
    """Convert frame to RGB for Gradio display."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def create_interface(debug: bool = False):
    """Create and launch the Gradio interface."""
    with gr.Blocks(title="Live Pose Detection") as demo:
        gr.Markdown("# 🏃 Live Pose Detection")
        
        with gr.Row():
            with gr.Column():
                # Camera input
                camera = gr.Video(
                    label="Webcam Feed",
                    mirror_webcam=True,
                    height=540,
                    width=960,
                    include_audio=False,
                    autoplay=True
                )
                
                # Controls
                with gr.Row():
                    cpu_only = gr.Checkbox(
                        label="CPU Only",
                        value=False,
                        info="Force CPU inference"
                    )
                    camera_index = gr.Number(
                        label="Camera Index",
                        value=0,
                        precision=0,
                        minimum=0,
                        maximum=10
                    )
                
                # Status indicators
                with gr.Row():
                    fps_display = gr.Textbox(
                        label="FPS",
                        value="0.0",
                        interactive=False
                    )
                    device_status = gr.Textbox(
                        label="Device",
                        value="Initializing...",
                        interactive=False
                    )
            
            # Output image
            output_image = gr.Image(
                label="Pose Detection",
                height=540,
                width=960
            )
        
        def process_video(camera_index, cpu_only):
            device = select_device(force_cpu=cpu_only)
            device_status = f"{device} {'⚡' if device == 'GPU' else ''}"
            
            for frame, fps in run_pose(
                cam_index=int(camera_index),
                use_gpu=(device == "GPU"),
                debug=debug
            ):
                yield process_frame(frame, fps), f"{fps:.1f}", device_status
        
        # Set up the video processing
        camera.change(
            process_video,
            inputs=[camera_index, cpu_only],
            outputs=[output_image, fps_display, device_status],
            every=1/30  # Update at 30 FPS
        )
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Pose Detection")
    parser.add_argument("--debug", action="store_true", help="Show OpenCV debug window")
    args = parser.parse_args()
    
    demo = create_interface(debug=args.debug)
    demo.launch(share=False) 