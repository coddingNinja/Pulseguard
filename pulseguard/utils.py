"""Shared helpers: paths, video reading, ROI cache."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("pulseguard")


def project_root() -> Path:
    """Folder that contains the `pulseguard/` package (the repo root)."""
    return Path(__file__).resolve().parents[1]


DEFAULT_CACHE_DIR = project_root() / "data" / "cache"
DEFAULT_OUTPUT_DIR = project_root() / "outputs"


# --------------------------------------------------------------------------
# Video reading
# --------------------------------------------------------------------------
def open_video(video_path) -> Tuple[cv2.VideoCapture, float]:
    """Open a video and return (capture, fps). Falls back to 30 fps if the
    file does not report a usable frame rate."""
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 1.0:
        log.warning("Video reports fps=%s; assuming 30 fps (%s)", fps, video_path)
        fps = 30.0
    return cap, fps


def iter_frames_rgb(video_path) -> Iterator[Tuple[int, np.ndarray, float]]:
    """Yield (frame_index, frame_rgb, fps) for every frame of the video."""
    cap, fps = open_video(video_path)
    try:
        i = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield i, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), fps
            i += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def cache_path(video_path, cache_dir: Optional[Path] = None) -> Path:
    """data/cache/<video_name>.npz (video_name = file name without extension)."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    return cache_dir / f"{Path(video_path).stem}.npz"


def save_cache(path: Path, rois: np.ndarray, fps: float, detection_rate: float,
               source: str, region_names) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rois=rois,
        fps=np.float64(fps),
        detection_rate=np.float64(detection_rate),
        source=np.str_(source),
        region_names=np.array(list(region_names)),
    )


def load_cache(path: Path):
    """Return the cache dict, or None if it does not exist / is unreadable."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return {
                "rois": z["rois"],
                "fps": float(z["fps"]),
                "detection_rate": float(z["detection_rate"]),
                "source": str(z["source"]),
                "region_names": [str(n) for n in z["region_names"]],
            }
    except Exception as exc:  # corrupted file: just recompute
        log.warning("Ignoring unreadable cache %s (%s)", path, exc)
        return None
