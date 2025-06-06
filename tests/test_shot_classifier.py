import sys
import types
from pathlib import Path

# Ensure src package is importable
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

# Provide minimal stubs so shot_detector imports without heavy dependencies
if 'torch' not in sys.modules:
    torch_stub = types.ModuleType('torch')
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules['torch'] = torch_stub

if 'ultralytics' not in sys.modules:
    class DummyYOLO:
        def __init__(self, *a, **kw):
            pass
        def predict(self, *a, **kw):
            return []
    ultralytics_stub = types.ModuleType('ultralytics')
    ultralytics_stub.YOLO = DummyYOLO
    sys.modules['ultralytics'] = ultralytics_stub

if 'cv2' not in sys.modules:
    sys.modules['cv2'] = types.ModuleType('cv2')

if 'numpy' not in sys.modules:
    numpy_stub = types.ModuleType('numpy')

    def array(x):
        return list(x)

    def polyfit(x, y, deg):
        return (0.0, 0.0, 0.0)

    numpy_stub.array = array
    numpy_stub.polyfit = polyfit
    sys.modules['numpy'] = numpy_stub

from shot_detector import ShotClassifier


def test_two_shots_classified():
    rim_center = (0, 0)
    rim_radius = 10
    classifier = ShotClassifier(rim_center, rim_radius)

    # Synthetic track data for two shots separated by a gap in frame numbers
    tracks = [
        # first shot
        {'frame': 0, 'x': 0, 'y': 0},
        {'frame': 1, 'x': 1, 'y': 1},
        {'frame': 2, 'x': 2, 'y': 2},
        {'frame': 3, 'x': 3, 'y': 3},
        # large gap before second shot
        {'frame': 20, 'x': 0, 'y': 0},
        {'frame': 21, 'x': 1, 'y': 1},
        {'frame': 22, 'x': 2, 'y': 2},
        {'frame': 23, 'x': 3, 'y': 3},
    ]

    result = classifier.classify(tracks, (rim_center, rim_radius))
    assert len(result) == 2
