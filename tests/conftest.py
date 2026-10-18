"""Shared test helpers: synthetic videos and a fake landmark tracker.

The real MediaPipe tracker needs a downloaded model file, so most tests use a
FakeTracker that returns fixed synthetic landmarks. That tests our own logic
(smoothing, masks, interpolation, caching) without needing a real face.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W, H = 160, 120


def make_video(path, n_frames=40, fps=30, color_fn=None):
    """Write a small MJPG .avi. color_fn(t) -> (R, G, B); default is flat grey-ish."""
    color_fn = color_fn or (lambda t: (200, 120, 80))
    path = str(path)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    assert vw.isOpened(), "could not open VideoWriter"
    for t in range(n_frames):
        r, g, b = color_fn(t)
        frame_bgr = np.empty((H, W, 3), np.uint8)
        frame_bgr[:] = (b, g, r)
        vw.write(frame_bgr)
    vw.release()
    return Path(path)


def fake_landmarks(seed=0):
    """468 random but fixed points inside an ellipse in the middle of the frame."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, 468)
    rad = np.sqrt(rng.uniform(0.05, 1.0, 468))
    x = W / 2 + 40 * rad * np.cos(ang)
    y = H / 2 + 45 * rad * np.sin(ang)
    return np.stack([x, y], axis=1)


class FakeTracker:
    """Returns fixed landmarks, or None for frames listed in `missing`."""

    def __init__(self, missing=(), all_missing=False, landmarks=None):
        self.missing = set(missing)
        self.all_missing = all_missing
        self.landmarks = fake_landmarks() if landmarks is None else landmarks
        self.calls = 0

    def detect(self, frame_rgb, timestamp_ms):
        i = self.calls
        self.calls += 1
        if self.all_missing or i in self.missing:
            return None
        return self.landmarks.copy()


@pytest.fixture
def tmp_video(tmp_path):
    return make_video(tmp_path / "clip.avi")
