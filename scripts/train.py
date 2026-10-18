"""Owner: Muhammad Hamza Tahir
Task ID: B4 / Train Classifier
Acceptance Criteria:
    - Trained on train split only (splits.csv).
    - Reports train & validation metrics (AUC, Accuracy, F1).
    - Saves trained model to models/classifier.pkl.
Can import:
    sklearn, xgboost, pandas, numpy, pulseguard.features, pulseguard.utils
TODO:
    1. Load extracted .npz files for train split clips.
    2. Compute 72-feature vectors using compute_features().
    3. Fit XGBoost / SVM classifier.
    4. Save trained weights to models/classifier.pkl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description="Model training script (Task B4)")
    ap.add_argument("--splits", default="data/splits.csv", help="path to splits.csv")
    ap.add_argument("--cache-dir", default="data/cache", help="directory with .npz caches")
    ap.add_argument("--out-model", default="models/classifier.pkl", help="output model path")
    args = ap.parse_args()

    print("=== Model Training Runner (Task B4 - Stub) ===")
    print(f"Splits:    {args.splits}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Output:    {args.out_model}")
    print("\nTODO: Muhammad Hamza Tahir to implement model training and export.")
    sys.exit(0)


if __name__ == "__main__":
    main()
