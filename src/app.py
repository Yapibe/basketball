"""
Gradio web interface for basketball shot analysis.
"""
import os
import gradio as gr
from shot_detector import RimDetector, BallTracker, ShotClassifier
import cv2
import tempfile

def analyze_video(video_path: str, cpu_only: bool) -> str:
    """Process uploaded video and return analysis results."""
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video file"
        
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
        # Stream results as they come in
        rim = detector.locate(frame_iter())
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        tracks = tracker.track(frame_iter())
        shots = classifier.classify(tracks, rim)
        
        results = []
        for shot in shots:
            result = f"Shot #{shot['id']} — {'MADE' if shot['made'] else 'MISS'} (flight {shot['t']:.2f}s)"
            results.append(result)
            
        return "\n".join(results)
        
    finally:
        cap.release()

def create_ui():
    """Create and launch Gradio interface."""
    with gr.Blocks(title="Shot Coach") as demo:
        gr.Markdown("# 🏀 Shot Coach")
        gr.Markdown("Upload a basketball video to analyze shot attempts.")
        
        with gr.Row():
            video_input = gr.Video(label="Upload Video")
            cpu_checkbox = gr.Checkbox(label="CPU Only", value=False)
            
        analyze_btn = gr.Button("Run Analysis")
        output = gr.Textbox(label="Results", lines=10)
        
        analyze_btn.click(
            fn=analyze_video,
            inputs=[video_input, cpu_checkbox],
            outputs=output
        )
        
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch() 