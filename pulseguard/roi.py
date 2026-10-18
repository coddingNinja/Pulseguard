"""A2: ROI extraction. 16 face regions -> mean RGB per region per frame.

Output of `extract_rois` is an array of shape [16, T, 3]:
    axis 0 = region (order = REGION_NAMES)
    axis 1 = frame
    axis 2 = R, G, B (0-255, float)

"r"/"l" in region names follow MediaPipe's convention: the SUBJECT's right/left.
On screen (un-mirrored video) the subject's right side appears on the image LEFT.

TODO(landmarks): the index lists below were written from knowledge of the
MediaPipe Face Mesh topology, not tuned on real footage. Run
`python scripts/draw_rois.py <video>` and check every mask sits on skin (no eyes,
lips, hair or background). Edit the *_R lists below if a region is off; the left
side is generated automatically by mirroring, so you only fix one side.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .tracking import N_LANDMARKS, LandmarkDetector, LandmarkSmoother, iter_tracked_frames
from .utils import cache_path, load_cache, open_video, save_cache

log = logging.getLogger("pulseguard")

# --------------------------------------------------------------------------
# Region definitions (edit here)
# --------------------------------------------------------------------------
# Left <-> right landmark pairs of the Face Mesh. Used to build the left-hand
# regions from the right-hand ones, so both sides are always symmetric.
_MIRROR_PAIRS = {
    # forehead / brows
    21: 251, 54: 284, 103: 332, 67: 297, 109: 338, 68: 298, 104: 333, 69: 299,
    108: 337, 71: 301, 70: 300, 63: 293, 105: 334, 66: 296, 107: 336, 55: 285,
    # cheeks
    116: 345, 117: 346, 118: 347, 119: 348, 101: 330, 50: 280, 123: 352,
    147: 376, 187: 411, 205: 425, 36: 266, 142: 371, 203: 423, 206: 426,
    207: 427, 227: 447, 137: 366, 177: 401, 215: 435, 213: 433, 192: 416,
    # nose
    129: 358, 49: 279, 98: 327,
    # mouth area / chin / jaw
    37: 267, 167: 393, 83: 313, 182: 406, 150: 379, 149: 378, 176: 400,
    148: 377,
}
_MIRROR = dict(_MIRROR_PAIRS)
_MIRROR.update({v: k for k, v in _MIRROR_PAIRS.items()})
# Landmarks on the vertical midline: they map to themselves.
_CENTER = {10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 2, 164, 0, 18, 200, 175}


def _mirror(indices: List[int]) -> List[int]:
    out = []
    for i in indices:
        if i in _CENTER:
            out.append(i)
        elif i in _MIRROR:
            out.append(_MIRROR[i])
        else:
            raise KeyError(f"Landmark {i} has no mirror entry; add it to _MIRROR_PAIRS or _CENTER")
    return out


# Right-hand (subject's right) definitions. Each list is turned into a polygon
# with the convex hull of its landmarks.
_FOREHEAD_R_OUTER = [21, 54, 103, 68, 71, 70, 63]
_FOREHEAD_R_INNER = [103, 67, 109, 10, 151, 9, 107, 66, 105, 104, 69, 108]
_CHEEK_R_UPPER = [116, 117, 118, 119, 101, 50, 123]
_CHEEK_R_LOWER = [101, 50, 205, 36, 142, 203, 206, 207, 187, 123, 147]
_CHEEK_R_OUTER = [227, 137, 177, 215, 213, 192, 147, 123, 116]
_CHIN_R = [83, 182, 150, 149, 176, 148, 200]

# Centre / symmetric regions.
_GLABELLA = [107, 9, 336, 8, 168, 55, 285]
_NOSE = [6, 197, 195, 5, 4, 1, 19, 129, 358, 98, 327, 49, 279]
_PHILTRUM = [2, 164, 167, 393, 37, 267]
_CHIN_CENTER = [18, 83, 313, 200, 175]

# Order here = order of axis 0 in the [16, T, 3] output. Do not reorder
# without telling the team (features / heatmap code depends on it).
REGIONS: "OrderedDict[str, List[int]]" = OrderedDict([
    ("forehead_r_outer", _FOREHEAD_R_OUTER),
    ("forehead_r_inner", _FOREHEAD_R_INNER),
    ("forehead_l_inner", _mirror(_FOREHEAD_R_INNER)),
    ("forehead_l_outer", _mirror(_FOREHEAD_R_OUTER)),
    ("glabella", _GLABELLA),
    ("nose", _NOSE),
    ("cheek_r_upper", _CHEEK_R_UPPER),
    ("cheek_r_lower", _CHEEK_R_LOWER),
    ("cheek_r_outer", _CHEEK_R_OUTER),
    ("cheek_l_upper", _mirror(_CHEEK_R_UPPER)),
    ("cheek_l_lower", _mirror(_CHEEK_R_LOWER)),
    ("cheek_l_outer", _mirror(_CHEEK_R_OUTER)),
    ("philtrum", _PHILTRUM),
    ("chin_center", _CHIN_CENTER),
    ("chin_r", _CHIN_R),
    ("chin_l", _mirror(_CHIN_R)),
])
REGION_NAMES: List[str] = list(REGIONS.keys())
N_REGIONS = len(REGIONS)

# Pull every polygon slightly toward its own centre so regions near the face
# edge do not pick up hair or background.
ROI_SHRINK = 0.9

assert N_REGIONS == 16, "the contract says 16 regions"
assert all(0 <= i < N_LANDMARKS for idx in REGIONS.values() for i in idx)


# --------------------------------------------------------------------------
# Geometry + colour averaging
# --------------------------------------------------------------------------
def region_polygons(landmarks: np.ndarray, shrink: float = ROI_SHRINK) -> List[np.ndarray]:
    """Return one int32 polygon [k, 2] (pixel coords) per region."""
    polys = []
    for idx in REGIONS.values():
        pts = landmarks[idx].astype(np.float32)
        hull = cv2.convexHull(pts).reshape(-1, 2)
        centre = hull.mean(axis=0)
        hull = centre + (hull - centre) * shrink
        polys.append(np.rint(hull).astype(np.int32))
    return polys


def region_means(frame_rgb: np.ndarray, landmarks: np.ndarray,
                 shrink: float = ROI_SHRINK) -> np.ndarray:
    """Mean R, G, B inside each region for one frame -> [16, 3] (NaN if empty)."""
    out = np.full((N_REGIONS, 3), np.nan)
    if landmarks is None or not np.all(np.isfinite(landmarks)):
        return out

    h, w = frame_rgb.shape[:2]
    polys = region_polygons(landmarks, shrink)

    # Work on a crop around the face: much cheaper than full-frame masks.
    allp = np.concatenate(polys)
    x0, y0 = max(int(allp[:, 0].min()), 0), max(int(allp[:, 1].min()), 0)
    x1, y1 = min(int(allp[:, 0].max()) + 1, w), min(int(allp[:, 1].max()) + 1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return out
    crop = np.ascontiguousarray(frame_rgb[y0:y1, x0:x1])

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for r, poly in enumerate(polys):
        mask[:] = 0
        cv2.fillConvexPoly(mask, poly - np.array([x0, y0], dtype=np.int32), 255)
        if cv2.countNonZero(mask) == 0:
            continue
        out[r] = cv2.mean(crop, mask=mask)[:3]
    return out


def interpolate_nans(rois: np.ndarray) -> np.ndarray:
    """Fill NaN frames of a [16, T, 3] array by linear interpolation over time.

    Frames before the first / after the last valid value take the nearest valid
    value. A trace that has no valid frame at all stays NaN.
    """
    rois = rois.copy()
    T = rois.shape[1]
    t = np.arange(T)
    for r in range(rois.shape[0]):
        for c in range(rois.shape[2]):
            y = rois[r, :, c]
            ok = np.isfinite(y)
            if ok.all() or not ok.any():
                continue
            rois[r, :, c] = np.interp(t, t[ok], y[ok])
    return rois


# --------------------------------------------------------------------------
# Visual check
# --------------------------------------------------------------------------
def draw_rois(frame_rgb: np.ndarray, landmarks: np.ndarray,
              show_indices: bool = False, alpha: float = 0.45,
              numbers: bool = True, legend: bool = True) -> np.ndarray:
    """Return a copy of the frame with the 16 region masks drawn and numbered."""
    img = frame_rgb.copy()
    overlay = img.copy()
    polys = region_polygons(landmarks)
    colours = []
    for r in range(N_REGIONS):
        hsv = np.uint8([[[int(180 * r / N_REGIONS), 220, 255]]])
        colours.append(tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]))
    for poly, col in zip(polys, colours):
        cv2.fillConvexPoly(overlay, poly, col)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    for r, (poly, col) in enumerate(zip(polys, colours)):
        cv2.polylines(img, [poly], True, col, 1, cv2.LINE_AA)
        if numbers:
            cx, cy = poly.mean(axis=0).astype(int)
            cv2.putText(img, str(r + 1), (cx - 6, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, str(r + 1), (cx - 6, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1, cv2.LINE_AA)
    if show_indices:
        for i, (x, y) in enumerate(landmarks):
            cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.2, (255, 255, 0), 1, cv2.LINE_AA)
    # legend
    if legend:
        for r, (name, col) in enumerate(zip(REGION_NAMES, colours)):
            y = 14 + 14 * r
            cv2.putText(img, f"{r + 1}: {name}", (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"{r + 1}: {name}", (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, col, 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    rois: np.ndarray            # [16, T, 3]
    fps: float
    detection_rate: float       # fraction of frames with a real face detection
    from_cache: bool
    region_names: List[str]


def extract_rois_ex(video_path, use_cache: bool = True, force: bool = False,
                    cache_dir=None, tracker: Optional[LandmarkDetector] = None,
                    smoother: Optional[LandmarkSmoother] = None) -> ExtractionResult:
    """Full version of extract_rois: also returns detection rate and cache info.

    The cache (data/cache/<video_name>.npz) does not record tracker settings,
    so pass force=True after changing the smoother or the region definitions.
    """
    video_path = Path(video_path)
    cpath = cache_path(video_path, cache_dir)
    source = str(video_path.resolve())

    if use_cache and not force:
        c = load_cache(cpath)
        if c and c["source"] == source and c["region_names"] == REGION_NAMES:
            log.info("Loaded ROI cache %s", cpath)
            return ExtractionResult(c["rois"], c["fps"], c["detection_rate"], True, REGION_NAMES)

    frames: List[np.ndarray] = []   # per-frame [16, 3]
    n_detected = 0
    cap, fps = open_video(video_path)
    cap.release()

    for tf in iter_tracked_frames(video_path, tracker, smoother):
        frames.append(region_means(tf.frame_rgb, tf.landmarks))
        n_detected += int(tf.detected)

    T = len(frames)
    if T == 0:
        raise ValueError(f"No frames could be read from {video_path}")

    rois = np.stack(frames, axis=1)          # [16, T, 3]
    detection_rate = n_detected / T
    rois = interpolate_nans(rois)
    if not np.isfinite(rois).any():
        log.warning("No face found in any frame of %s", video_path)

    if use_cache:
        save_cache(cpath, rois, fps, detection_rate, source, REGION_NAMES)
    return ExtractionResult(rois, fps, detection_rate, False, REGION_NAMES)


def extract_rois(video_path, use_cache: bool = True, force: bool = False,
                 cache_dir=None, tracker: Optional[LandmarkDetector] = None
                 ) -> Tuple[np.ndarray, float]:
    """Contract function. Returns (rois [16, T, 3], fps)."""
    r = extract_rois_ex(video_path, use_cache=use_cache, force=force,
                        cache_dir=cache_dir, tracker=tracker)
    return r.rois, r.fps
