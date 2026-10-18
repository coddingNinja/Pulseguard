"""A1: face tracking + landmark stabilisation.

Uses MediaPipe's FaceLandmarker (the "Tasks" API). The older
`mp.solutions.face_mesh` API was removed from recent MediaPipe releases
(it is gone in 0.10.32), so the Tasks API is the one that works going forward.
Both give the same 468-point Face Mesh topology that roi.py relies on.

The Tasks API needs a model file (`face_landmarker.task`). `ensure_model()`
downloads it to models/ the first time it is needed.
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol

import numpy as np

from .utils import iter_frames_rgb, project_root

log = logging.getLogger("pulseguard")

N_LANDMARKS = 468  # FaceLandmarker also returns 10 iris points; we drop them.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = project_root() / "models" / "face_landmarker.task"


# --------------------------------------------------------------------------
# Model file
# --------------------------------------------------------------------------
def ensure_model(model_path=None) -> Path:
    """Return the path to face_landmarker.task, downloading it if missing."""
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading FaceLandmarker model to %s", model_path)
    tmp = model_path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(model_path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the FaceLandmarker model ({exc}).\n"
            f"Download it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {model_path}"
        ) from exc
    return model_path


# --------------------------------------------------------------------------
# Tracker (thin wrapper around MediaPipe)
# --------------------------------------------------------------------------
class LandmarkDetector(Protocol):
    """Anything with this method can be used as a tracker (tests use fakes)."""

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        """Return landmarks [468, 2] in pixel coordinates, or None if no face."""


class FaceTracker:
    """MediaPipe FaceLandmarker in VIDEO mode (tracks across frames)."""

    def __init__(self, model_path=None, min_detection_conf: float = 0.5,
                 min_presence_conf: float = 0.5, min_tracking_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        path = ensure_model(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        # VIDEO mode needs strictly increasing timestamps.
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms
        h, w = frame_rgb.shape[:2]
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb),
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks:
            return None
        pts = result.face_landmarks[0][:N_LANDMARKS]
        return np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float64)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# Stabilisation
# --------------------------------------------------------------------------
class LandmarkSmoother:
    """Smooths landmarks over time and bridges short no-face gaps.

    - Exponential moving average: smoothed = alpha*new + (1-alpha)*previous.
      Higher alpha = less smoothing, less lag.
    - If the face jumps a long way in one frame (new person / re-detection),
      the average is reset instead of smearing the face across the jump.
    - If no face is found, the last good landmarks are held for up to
      `max_gap` frames. After that the frame is reported as invalid (None).
    """

    def __init__(self, alpha: float = 0.6, max_gap: int = 10, jump_frac: float = 0.25):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.max_gap = max_gap
        self.jump_frac = jump_frac
        self._prev: Optional[np.ndarray] = None
        self._gap = 0

    def reset(self) -> None:
        self._prev = None
        self._gap = 0

    def update(self, raw: Optional[np.ndarray]):
        """Feed one frame. Returns (landmarks_or_None, detected).

        `detected` is True only when the face was really found in this frame
        (not when the landmarks were carried forward).
        """
        if raw is None:
            self._gap += 1
            if self._prev is not None and self._gap <= self.max_gap:
                return self._prev.copy(), False
            self._prev = None
            return None, False

        self._gap = 0
        if self._prev is None:
            self._prev = raw.copy()
        else:
            face_w = np.ptp(self._prev[:, 0]) + 1e-6
            shift = np.mean(np.linalg.norm(raw - self._prev, axis=1))
            if shift > self.jump_frac * face_w:
                self._prev = raw.copy()
            else:
                self._prev = self.alpha * raw + (1.0 - self.alpha) * self._prev
        return self._prev.copy(), True


# --------------------------------------------------------------------------
# Video-level API
# --------------------------------------------------------------------------
@dataclass
class TrackedFrame:
    index: int
    frame_rgb: np.ndarray
    landmarks: Optional[np.ndarray]  # [468, 2] pixels, or None if invalid
    detected: bool                    # True = real detection this frame


def iter_tracked_frames(video_path, tracker: Optional[LandmarkDetector] = None,
                        smoother: Optional[LandmarkSmoother] = None
                        ) -> Iterator[TrackedFrame]:
    """Run the tracker + smoother over a video, one frame at a time.

    Frames are yielded as they are processed, so callers (roi.py) can use the
    pixels without decoding the video a second time.
    """
    own_tracker = tracker is None
    if own_tracker:
        tracker = FaceTracker()
    smoother = smoother or LandmarkSmoother()
    prev_ts = -1
    try:
        for i, frame_rgb, fps in iter_frames_rgb(video_path):
            ts = max(int(round(i * 1000.0 / fps)), prev_ts + 1)
            prev_ts = ts
            raw = tracker.detect(frame_rgb, ts)
            landmarks, detected = smoother.update(raw)
            yield TrackedFrame(i, frame_rgb, landmarks, detected)
    finally:
        if own_tracker and hasattr(tracker, "close"):
            tracker.close()


@dataclass
class TrackResult:
    landmarks: np.ndarray   # [T, 468, 2], NaN where invalid
    detected: np.ndarray    # [T] bool, real detections only
    valid: np.ndarray       # [T] bool, detected OR carried forward
    fps: float
    detection_rate: float   # fraction of frames with a real detection


def track_video(video_path, tracker: Optional[LandmarkDetector] = None,
                smoother: Optional[LandmarkSmoother] = None) -> TrackResult:
    """Track the face through a whole video and return landmarks per frame."""
    from .utils import open_video

    cap, fps = open_video(video_path)
    cap.release()

    lms, det, val = [], [], []
    for tf in iter_tracked_frames(video_path, tracker, smoother):
        if tf.landmarks is None:
            lms.append(np.full((N_LANDMARKS, 2), np.nan))
            val.append(False)
        else:
            lms.append(tf.landmarks)
            val.append(True)
        det.append(tf.detected)

    T = len(lms)
    detected = np.array(det, dtype=bool)
    return TrackResult(
        landmarks=np.stack(lms) if T else np.zeros((0, N_LANDMARKS, 2)),
        detected=detected,
        valid=np.array(val, dtype=bool),
        fps=fps,
        detection_rate=float(detected.mean()) if T else 0.0,
    )
