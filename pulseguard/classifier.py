"""Owner: Muhammad Hamza Tahir
Task ID: B4 / Baseline classifier
Acceptance Criteria:
    - Trained on train split only (splits.csv, by identity).
    - AUC, accuracy, F1 on held-out test split.
    - Saves trained model artifact to models/classifier.pkl.
    - Baseline: SNR threshold rule, then XGBoost / SVM.
Can import:
    sklearn, xgboost, numpy, pandas, pulseguard.features
TODO:
    1. Implement SNR threshold rule baseline.
    2. Train XGBoost classifier on 72-feature vectors from scripts/train.py.
    3. Save model with joblib / pickle.
    4. Implement predict_proba / predict_features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np


class DeepfakeClassifier:
    """Trained classifier predicting real vs synthetic video using biological rPPG features."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else None
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train classifier on feature matrix X [N, 72] and labels y [N] (1=real, 0=fake)."""
        raise NotImplementedError("Task B4: Muhammad Hamza Tahir to implement model training")

    def load(self, model_path: Union[str, Path]):
        """Load trained model weights from disk."""
        raise NotImplementedError("Task B4: Muhammad Hamza Tahir to implement model loading")

    def save(self, model_path: Union[str, Path]):
        """Save model artifact to disk."""
        raise NotImplementedError("Task B4: Muhammad Hamza Tahir to implement model saving")

    def predict_features(self, feature_vector: np.ndarray) -> dict:
        """Predict real vs fake score from a 72-element feature vector."""
        raise NotImplementedError("Task B4: Muhammad Hamza Tahir to implement feature prediction")
