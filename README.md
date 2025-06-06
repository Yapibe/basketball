# Shot Coach

A computer vision-based basketball shot analysis system that automatically detects and classifies shot attempts.

## Features

- Automatic rim detection using Hough transform or YOLO
- Ball tracking using YOLOv8
- Shot classification (Made/Missed)
- Real-time feedback via Gradio interface
- GPU acceleration support (optional)

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd shot_coach
```

2. Create and activate virtual environment:
```bash
uv venv
# On Windows:
.venv/Scripts/activate
# On Unix/MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip sync
```

## Usage

### GUI Mode
```bash
python -m src.app
```

### Command Line Mode
```bash
python -m src.shot_detector path/to/video.mp4
```

## Recording Tips

- Use a tripod at ~1.6m height
- Position camera at 45° angle from the basket
- Ensure good lighting
- Avoid orange jerseys (may interfere with ball detection)
- Record in 1080p @ 30fps

## System Requirements

- Python 3.12+
- OpenCV
- CUDA-capable GPU (optional, RTX 3050 or better recommended)
- 8GB+ RAM

## License

MIT 