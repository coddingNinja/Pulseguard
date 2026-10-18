"""Test A1 + A2 live on your webcam.

Usage:
    python scripts/webcam_roi.py                  # camera 0, runs until you press q
    python scripts/webcam_roi.py --seconds 30     # stop by itself after 30 s
    python scripts/webcam_roi.py --mirror         # preview like a mirror
    python scripts/webcam_roi.py --camera 1       # another camera

In the window:  q or Esc = quit,  s = save a snapshot with the masks drawn.
At the end it prints the detection rate and saves the 16 colour traces to
data/cache/webcam_<time>.npz (same format as the video cache).
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pulseguard.live import run_live


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--source", default=None,
                    help="use a video file instead of the camera (for testing)")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--mirror", action="store_true", help="mirror the preview (display only)")
    ap.add_argument("--no-save", action="store_true", help="do not save the traces")
    ap.add_argument("--warmup", type=int, default=15,
                    help="camera frames to discard at the start (default 15)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--model", default=None, help="path to face_landmarker.task")
    ap.add_argument("--method", choices=["pos", "chrom"], default="pos",
                    help="pulse recovery method: pos or chrom (default pos)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    tracker = None
    if args.model:
        from pulseguard.tracking import FaceTracker
        tracker = FaceTracker(args.model)

    print("Starting. Sit about 50 cm from the camera with light on your face.")
    print("Keep still for the first test. Press q in the window to finish.\n")
    res = run_live(source=args.source if args.source else args.camera, tracker=tracker,
                   seconds=args.seconds, mirror=args.mirror, warmup=args.warmup,
                   save=not args.no_save, width=args.width, height=args.height,
                   method=args.method)

    print("\n=== Summary ===")
    print(f"frames         : {res.n_frames}  ({res.duration_s:.1f} s)")
    print(f"measured fps   : {res.fps:.1f}")
    print(f"detection rate : {res.detection_rate * 100:.1f}%  "
          f"({'PASS' if res.detection_rate >= 0.95 else 'below the 95% target'})")
    print(f"rois shape     : {res.rois.shape}   NaN values: {int(np.isnan(res.rois).sum())}")
    if res.saved_to:
        print(f"traces saved   : {res.saved_to}")


if __name__ == "__main__":
    main()
