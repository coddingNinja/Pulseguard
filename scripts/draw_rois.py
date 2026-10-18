"""Save a frame with the 16 ROI masks drawn on it (the A2 visual check).

Usage:
    python scripts/draw_rois.py data/raw/my_clip.mp4              # frame 30
    python scripts/draw_rois.py data/raw/my_clip.mp4 --frame 100
    python scripts/draw_rois.py data/raw/my_clip.mp4 --indices    # also number every landmark

Output: outputs/<video_name>_rois.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from pulseguard.roi import draw_rois
from pulseguard.tracking import iter_tracked_frames
from pulseguard.utils import DEFAULT_OUTPUT_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=30,
                    help="frame to draw (default 30, so the smoother is warmed up)")
    ap.add_argument("--indices", action="store_true",
                    help="also print the landmark index next to every point (for tuning regions)")
    ap.add_argument("--out", default=None, help="output png path")
    args = ap.parse_args()

    chosen = None
    for tf in iter_tracked_frames(args.video):
        if tf.landmarks is not None:
            chosen = tf
        if tf.index >= args.frame and chosen is not None:
            break
    if chosen is None:
        sys.exit("No face found in the video, nothing to draw.")

    img = draw_rois(chosen.frame_rgb, chosen.landmarks, show_indices=args.indices)
    out = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / f"{Path(args.video).stem}_rois.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved {out}  (frame {chosen.index})")


if __name__ == "__main__":
    main()
