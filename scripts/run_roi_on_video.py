"""Extract the 16 ROI colour traces from one video and print a summary.

Usage:
    python scripts/run_roi_on_video.py data/raw/my_clip.mp4
    python scripts/run_roi_on_video.py data/raw/my_clip.mp4 --force
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pulseguard.roi import REGION_NAMES, extract_rois_ex


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="path to a video file")
    ap.add_argument("--force", action="store_true", help="ignore the cache and recompute")
    ap.add_argument("--no-cache", action="store_true", help="do not read or write the cache")
    ap.add_argument("--model", default=None, help="path to face_landmarker.task")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tracker = None
    if args.model:
        from pulseguard.tracking import FaceTracker
        tracker = FaceTracker(args.model)

    t0 = time.time()
    res = extract_rois_ex(args.video, use_cache=not args.no_cache, force=args.force,
                          tracker=tracker)
    dt = time.time() - t0

    rois = res.rois
    T = rois.shape[1]
    print(f"video          : {args.video}")
    print(f"rois shape     : {rois.shape}  (regions, frames, RGB)")
    print(f"fps            : {res.fps:.2f}   duration: {T / res.fps:.1f} s")
    print(f"detection rate : {res.detection_rate * 100:.1f}%  "
          f"({'PASS' if res.detection_rate >= 0.95 else 'below the 95% target'})")
    print(f"NaN values     : {int(np.isnan(rois).sum())}")
    print(f"source         : {'cache' if res.from_cache else f'computed in {dt:.1f}s'}")
    print("\nmean RGB per region:")
    for name, m in zip(REGION_NAMES, np.nanmean(rois, axis=1)):
        print(f"  {name:<18s} R={m[0]:6.1f}  G={m[1]:6.1f}  B={m[2]:6.1f}")


if __name__ == "__main__":
    main()
