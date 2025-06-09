# Shot Coach Prototype

End‑to‑end demo that classifies basketball shots (MADE / MISSED) from a single 30 fps, 1080p clip.  
Phase 1 keeps the architecture open for pose‑quality feedback in Phase 2.

## Quick Start (Windows 11)

1. **Clone & install**
   ```bash
   git clone https://github.com/your‑org/shot_coach.git
   cd shot_coach
   uv venv
   uv pip sync
   ```

2. **Run the web UI**
   ```bash
   python src/app.py
   ```
   Open the Gradio link in your browser, upload a clip, and watch the console stream results.

3. **Headless CLI**
   ```bash
   python -m shot_detector path/to/video.mp4  # add --cpu-only if no GPU
   ```

## Recording Tips

* Place a tripod ~1.6 m high at a *45° side angle* (between baseline & free‑throw line).  
  This yields a clear circular rim for Hough detection.
* Indoor lighting: aim for ≥1/250 s shutter to avoid blur.
* Orange shirts can confuse HSV filters – YOLO fallback is enabled.

### Calibration fallback
If automatic rim detection fails, edit `src/shot_detector.py` ► `RimDetector.locate` and set
`self._cache = ((x, y), r)` from a frame grab. TODO: add interactive click utility.

## Performance Notes

* Prototype defaults to **CPU** but switches to **GPU** automatically via `torch.cuda.is_available()`.
* On low‑end laptops, you can down‑scale frames:
  ```python
  frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
  ```
  Keep ≥15 fps processing for smooth tracking.

## Future Work  <!-- TODOs -->

* MediaPipe Pose integration for shooter biomechanics.
* Gemini / Vertex AI feedback generator.
* Multiple shooters, side‑by‑side comparisons.
* Better shot segmentation (Kalman filter, SORT trackers).

## Sample Clip  (optional)

Download a 5‑second test clip (public domain)
<https://files.sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4>

## Roboflow Shot Detection

An optional helper, `roboflow_detector.py`, demonstrates how to run
Roboflow models. Install the extra dependency and run:

```bash
uv pip install inference-sdk
python -m roboflow_detector path/to/image.jpg
```

Set the environment variable `ROBOFLOW_API_KEY` to use your own API key
instead of the built in demo key.

# Live Pose Detection

Real-time pose detection using MediaPipe and Gradio, with GPU acceleration support.

## Features

- Real-time pose detection (≥15 FPS on CPU)
- Full 33-landmark skeleton overlay
- GPU acceleration support (RTX 3050 compatible)
- Live web interface with Gradio
- FPS monitoring and device status
- Debug mode with OpenCV window

## Requirements

- Python 3.10+
- Webcam
- NVIDIA GPU (optional, for acceleration)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd live_pose
```

2. Create and activate virtual environment with uv:
```bash
uv venv
```

3. Install dependencies:
```bash
uv pip sync
```

## Usage

Run the application:
```bash
python src/app.py
```

For debug mode (shows OpenCV window):
```bash
python src/app.py --debug
```

## Interface Controls

- **Camera Index**: Select which camera to use (default: 0)
- **CPU Only**: Force CPU inference (uncheck to use GPU if available)
- **FPS**: Current frames per second
- **Device**: Shows current inference device (CPU/GPU)

## Future Extensions

The code is structured to easily add:
- Shot classification logic
- Vertex AI integration for coaching feedback
- Additional pose analysis features

## Notes

- Uses DirectShow backend on Windows for better performance
- Frame size is optimized to 960×540 for speed
- MediaPipe Pose model complexity is set to "Heavy" for best accuracy
