# Shot Coach Prototype

End‑to‑end demo that classifies basketball shots (MADE / MISSED) from a single 30 fps, 1080p clip.  
Phase 1 keeps the architecture open for pose‑quality feedback in Phase 2.

## Quick Start (Windows 11)

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

## Recording Tips

* Place a tripod **~1.6 m** high at a *45° side angle* (between baseline & free‑throw line).  
  This yields a clear circular rim for Hough detection.
* Indoor lighting: aim for ≥1/250 s shutter to avoid blur.
* Orange shirts can confuse HSV filters – YOLO fallback is enabled.

### Calibration fallback
If automatic rim detection fails, edit `src/shot_detector.py` ► `RimDetector.locate` and set
`self._cache = ((x, y), r)` from a frame grab. TODO: add interactive click utility.

## Performance Notes

* Prototype defaults to **CPU** but switches to **GPU** automatically via `torch.cuda.is_available()`.
* On low‑end laptops, you can down‑scale frames:
  ```python
  frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
  ```
  Keep ≥15 fps processing for smooth tracking.

## Future Work  <!-- TODOs -->

* MediaPipe Pose integration for shooter biomechanics.
* Gemini / Vertex AI feedback generator.
* Multiple shooters, side‑by‑side comparisons.
* Better shot segmentation (Kalman filter, SORT trackers).

## Sample Clip  (optional)

Download a 5‑second test clip (public domain)  
<https://files.sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4>