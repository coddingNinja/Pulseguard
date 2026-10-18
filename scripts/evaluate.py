"""Owner: Muhammad Hamza Tahir
Task ID: B5 / Cross-dataset & per-quality evaluation
Acceptance Criteria:
    - Train on one dataset, test on the other (e.g., train on FF++, test on Celeb-DF).
    - Table of metrics broken down by compression level (raw, c23, c40).
    - Generates ROC plot in results/roc_curve.png.
Can import:
    sklearn.metrics, matplotlib, pandas, numpy, pulseguard.classifier
TODO:
    1. Load test split features.
    2. Compute AUC, precision, recall, confusion matrix.
    3. Generate breakdown by compression level.
    4. Save metrics table to results/eval_summary.csv and ROC curve to results/roc.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description="Model evaluation script (Task B5)")
    ap.add_argument("--model", default="models/classifier.pkl", help="path to trained model")
    ap.add_argument("--splits", default="data/splits.csv", help="path to splits.csv")
    ap.add_argument("--cache-dir", default="data/cache", help="directory with .npz caches")
    ap.add_argument("--out-dir", default="results", help="directory for evaluation outputs")
    args = ap.parse_args()

    print("=== Model Evaluation Runner (Task B5 - Stub) ===")
    print(f"Model:   {args.model}")
    print(f"Splits:  {args.splits}")
    print(f"Out dir: {args.out_dir}")
    print("\nTODO: Muhammad Hamza Tahir to implement evaluation metrics and ROC plots.")
    sys.exit(0)


if __name__ == "__main__":
    main()
