"""Live webcam mode: run A1 (tracking) + A2 (ROI extraction) on a camera feed.

This is the same pipeline as extract_rois, but frames come straight from the
webcam and you see the result live: the 16 masks on your face, whether the face
is being tracked, and a rolling plot of the mean green value of your cheeks.

`source` can be a camera index (0, 1, ...) or a video file path. The file option
exists so the code can be tested without a camera.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .pulse import recover_pulse
from .roi import N_REGIONS, REGION_NAMES, draw_rois, interpolate_nans, region_means
from .tracking import FaceTracker, LandmarkDetector, LandmarkSmoother
from .utils import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, save_cache

log = logging.getLogger("pulseguard")

WINDOW = "PulseGuard live  (q = quit, s = snapshot)"
_CHEEKS = [REGION_NAMES.index(n) for n in ("cheek_r_upper", "cheek_l_upper")]
CAMERA_HINT = (
    "Could not open the camera. On macOS: System Settings > Privacy & Security > "
    "Camera, and switch on the app you run this from (Terminal, iTerm, VS Code...). "
    "Then quit and reopen that app. Also close other apps using the camera "
    "(Zoom, FaceTime, browser tabs). Try --camera 1 if you have several cameras."
)


@dataclass
class LiveResult:
    rois: np.ndarray            # [16, T, 3], gaps interpolated
    fps: float                  # measured frames per second
    detection_rate: float       # fraction of frames with a real face detection
    n_frames: int
    duration_s: float
    saved_to: Optional[Path]    # npz written at the end (None if save=False)


def open_source(source: Union[int, str], width: int = 640, height: int = 480):
    """Open a camera (int or digit string) or a video file. Returns (cap, is_camera)."""
    is_cam = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
    cap = cv2.VideoCapture(int(source) if is_cam else str(source))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(CAMERA_HINT if is_cam else f"Could not open video: {source}")
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap, is_cam


def _draw_trace(img_rgb: np.ndarray, values, label: str,
                y_offset: int = 0, color=(80, 255, 80)) -> None:
    """Small rolling plot along the bottom of the frame (drawn in place)."""
    h, w = img_rgb.shape[:2]
    bh = max(42, h // 9)
    x0, y0, x1, y1 = 8, h - bh - 8 - y_offset, w - 8, h - 8 - y_offset
    box = img_rgb[y0:y1, x0:x1]
    box[:] = (box * 0.35).astype(np.uint8)          # darken the background
    cv2.putText(img_rgb, label, (x0 + 4, y0 + 13), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (255, 255, 255), 1, cv2.LINE_AA)
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return
    v = np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok])
    lo, hi = v.min(), v.max()
    span = max(hi - lo, 1e-6)
    xs = np.linspace(x0 + 2, x1 - 2, len(v))
    ys = y1 - 4 - (v - lo) / span * (bh - 20)
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_rgb, [pts], False, color, 1, cv2.LINE_AA)


def _hud(img_rgb, detected_now: bool, det_rate: float, fps: float, elapsed: float) -> None:
    ok_col, bad_col = (60, 255, 60), (255, 60, 60)
    txt = "FACE TRACKED" if detected_now else "NO FACE"
    col = ok_col if detected_now else bad_col
    h, w = img_rgb.shape[:2]
    bh = max(42, h // 9)
    line = f"{txt} | detect {det_rate * 100:5.1f}% | {fps:4.1f} fps | {elapsed:5.1f}s"
    y_pos = max(20, h - 8 - 2 * (bh + 6) - 6)
    cv2.putText(img_rgb, line, (8, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_rgb, line, (8, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def run_live(source: Union[int, str] = 0, tracker: Optional[LandmarkDetector] = None,
             seconds: Optional[float] = None, display: bool = True, mirror: bool = False,
             warmup: int = 15, save: bool = True, cache_dir=None, output_dir=None,
             width: int = 640, height: int = 480,
             max_frames: Optional[int] = None,
             method: str = "pos") -> LiveResult:
    """Run tracking + ROI extraction on a live source until 'q', `seconds`, or the end.

    warmup: frames thrown away at the start of a camera feed (auto-exposure
            settles; the first frames are often dark).
    mirror: show the preview like a mirror (display only; the data is not flipped).
    method: pulse recovery method ("pos" or "chrom").
    """
    cap, is_cam = open_source(source, width, height)
    own_tracker = tracker is None
    if own_tracker:
        tracker = FaceTracker()
    smoother = LandmarkSmoother()
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR

    means, times, detected = [], [], []
    green_hist: deque = deque(maxlen=300)
    pulse_hist: deque = deque(maxlen=300)
    current_bpm = np.nan
    pulse_label = f"pulse ({method.upper()}, band-passed) [warming up]"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        if is_cam:
            for _ in range(warmup):
                cap.read()
        t0 = time.perf_counter()
        frame_count = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                if not means and is_cam:
                    raise RuntimeError("Camera opened but gave no frames. " + CAMERA_HINT)
                break
            now = time.perf_counter() - t0
            if seconds is not None and now >= seconds:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw = tracker.detect(rgb, int(now * 1000))
            landmarks, det = smoother.update(raw)
            m = region_means(rgb, landmarks) if landmarks is not None else \
                np.full((N_REGIONS, 3), np.nan)
            means.append(m)
            times.append(now)
            detected.append(det)
            frame_count += 1
            cheek_g = m[_CHEEKS, 1]
            green_hist.append(float(np.mean(cheek_g)) if np.isfinite(cheek_g).any() else np.nan)

            n = len(times)
            fps_now = (n - 1) / (times[-1] - times[0]) if n > 1 and times[-1] > times[0] else 30.0

            # Keep rolling buffer of last ~12 s
            buf_frames = max(30, int(round(12.0 * fps_now)))
            min_8s_frames = max(20, int(round(8.0 * fps_now)))

            # Update pulse recovery every ~5 frames once at least 8 s of data exist
            if frame_count % 5 == 0 and n >= min_8s_frames:
                window_means = means[-buf_frames:]
                rois_window = interpolate_nans(np.stack(window_means, axis=1))
                if np.isfinite(rois_window).any() and fps_now > 8.5:
                    try:
                        rec = recover_pulse(rois_window, fps_now, method=method)
                        overall_sig = rec["overall"]
                        if np.isfinite(overall_sig).all():
                            pulse_hist.clear()
                            pulse_hist.extend(overall_sig.tolist())
                            current_bpm = rec["bpm"]
                    except Exception as exc:
                        log.debug("Pulse recovery error: %s", exc)

                if now >= 10.0 and np.isfinite(current_bpm):
                    pulse_label = f"pulse ({method.upper()}, band-passed) | BPM ~ {current_bpm:.0f} (est)"
                else:
                    pulse_label = f"pulse ({method.upper()}, band-passed) [warming up]"

            if display:
                vis = draw_rois(rgb, landmarks, numbers=not mirror, legend=not mirror) \
                    if landmarks is not None else rgb.copy()
                if mirror:
                    vis = np.ascontiguousarray(vis[:, ::-1])

                h = vis.shape[0]
                bh = max(42, h // 9)

                _hud(vis, det, float(np.mean(detected)), fps_now, now)
                # Bottom plot: raw green trace of cheeks
                _draw_trace(vis, green_hist, "mean green of upper cheeks (last ~10 s)",
                            y_offset=0, color=(80, 255, 80))
                # Top plot: recovered band-passed pulse waveform
                _draw_trace(vis, pulse_hist, pulse_label,
                            y_offset=bh + 6, color=(255, 120, 80))

                cv2.imshow(WINDOW, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s") and landmarks is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    snap = out_dir / f"webcam_snapshot_{stamp}_{len(times)}.png"
                    cv2.imwrite(str(snap), cv2.cvtColor(draw_rois(rgb, landmarks), cv2.COLOR_RGB2BGR))
                    print(f"Saved {snap}")
            if max_frames is not None and len(means) >= max_frames:
                break
    finally:
        cap.release()
        if own_tracker and hasattr(tracker, "close"):
            tracker.close()
        if display:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    T = len(means)
    if T == 0:
        raise RuntimeError("No frames were captured.")
    duration = times[-1] - times[0] if T > 1 else 0.0
    fps = (T - 1) / duration if duration > 0 else 30.0
    rois = interpolate_nans(np.stack(means, axis=1))
    det_rate = float(np.mean(detected))

    saved = None
    if save:
        saved = (Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR) / f"webcam_{stamp}.npz"
        save_cache(saved, rois, fps, det_rate, f"webcam:{source}", REGION_NAMES)
    return LiveResult(rois, fps, det_rate, T, duration, saved)
