"""Owner: Muhammad Hamza Tahir
Task ID: B3 / Batch extraction runner
Acceptance Criteria:
    - Runs the whole subset unattended (parallel, cached, resumable).
    - Skips finished clips already present in data/cache/.
    - Produces a failure log for clips with no face.
Can import:
    pulseguard.roi, pulseguard.utils, multiprocessing, pandas
TODO:
    1. Parse data/splits.csv to get video paths.
    2. Dispatch extract_rois_ex across multiple workers (multiprocessing.Pool).
    3. Record extraction status and save log to results/batch_extract_log.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description="Batch ROI extraction runner (Task B3)")
    ap.add_argument("--splits", default="data/splits.csv", help="path to splits.csv")
    ap.add_argument("--data-dir", default="data/raw", help="root directory of raw videos")
    ap.add_argument("--workers", type=int, default=4, help="number of parallel worker processes")
    ap.add_argument("--force", action="store_true", help="recompute existing caches")
    args = ap.parse_args()

    print("=== Batch Extraction Runner (Task B3 - Stub) ===")
    print(f"Splits file: {args.splits}")
    print(f"Data dir:    {args.data_dir}")
    print(f"Workers:     {args.workers}")
    print("\nTODO: Muhammad Hamza Tahir to implement multiprocessing extraction loop.")
    sys.exit(0)


if __name__ == "__main__":
    main()
