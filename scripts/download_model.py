"""Download the MediaPipe FaceLandmarker model to models/face_landmarker.task.

The tracker does this automatically on first use; run this script if you want
to do it up front (or to see a clear error if your network blocks it).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pulseguard.tracking import ensure_model

if __name__ == "__main__":
    print("Model ready at:", ensure_model())
