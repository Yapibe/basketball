import os
import gradio as gr
import cv2
from shot_detector import RimDetector, BallTracker, ShotClassifier
import torch

def analyze_video(video_path: str, use_cpu: bool) -> str:
    """Process video and return shot analysis results."""
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
    print(f"Using device: {device}")
    
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
        return "Error: Could not detect rim. Please try a different video angle."
    
    # Track ball
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ball_tracker = BallTracker()
    tracks = ball_tracker.track(frame_generator())
    
    # Classify shots
    classifier = ShotClassifier(rim_center, rim_radius)
    shots = classifier.classify(tracks)
    
    # Format results
    results = []
    for shot in shots:
        results.append(f"Shot #{shot['id']} — {'MADE' if shot['made'] else 'MISS'} (flight {shot['t']:.2f}s)")
    
    cap.release()
    return "\n".join(results)

# Create Gradio interface
with gr.Blocks(title="Basketball Shot Analysis") as demo:
    gr.Markdown("# Basketball Shot Analysis")
    gr.Markdown("""
    Upload a basketball video to analyze shot attempts.
    - Ideal recording angle: 45° side view
    - Camera height: ~1.6m
    - Avoid orange jerseys (may affect ball detection)
    """)
    
    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        use_cpu = gr.Checkbox(label="CPU Only", value=False)
    
    analyze_btn = gr.Button("Analyze Shots")
    output = gr.Textbox(label="Analysis Results", lines=10)
    
    analyze_btn.click(
        fn=analyze_video,
        inputs=[video_input, use_cpu],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch() 