"""Tests for pulseguard.coherence (A6: Cross-region coherence)."""

import numpy as np
import pytest

from pulseguard.coherence import coherence_features, coherence_matrix


def test_coherence_matrix_identical_signals():
    fps = 30.0
    T = 300
    t = np.arange(T) / fps
    base_sig = np.sin(2 * np.pi * 1.2 * t)

    # All 16 regions identical
    signals = np.tile(base_sig, (16, 1))
    mat = coherence_matrix(signals)

    assert mat.shape == (16, 16)
    # Off-diagonals and diagonals should all be ~1.0
    assert np.allclose(mat, 1.0, atol=1e-4)

    signal_dict = {"signals": signals, "fps": fps, "bpm": 72.0}
    feats = coherence_features(signal_dict)
    assert abs(feats["coherence_mean"] - 1.0) < 1e-3
    assert feats["coherence_std"] < 1e-3
    assert abs(feats["coherence_min"] - 1.0) < 1e-3
    assert feats["bpm_agreement_ratio"] == 1.0


def test_coherence_matrix_independent_noise():
    rng = np.random.default_rng(123)
    T = 1000
    # 16 independent noise sequences
    signals = rng.normal(0, 1.0, (16, T))
    mat = coherence_matrix(signals)

    # Diagonals are 1.0
    diag = np.diag(mat)
    assert np.allclose(diag, 1.0)

    # Off-diagonal should be near 0
    triu_idx = np.triu_indices(16, k=1)
    off_diag = mat[triu_idx]
    assert np.mean(np.abs(off_diag)) < 0.15


def test_coherence_nan_and_constant_regions():
    fps = 30.0
    T = 300
    t = np.arange(T) / fps
    signals = np.tile(np.sin(2 * np.pi * 1.2 * t), (16, 1))

    # Region 0 has NaN, Region 1 is constant
    signals[0] = np.nan
    signals[1] = 0.0

    mat = coherence_matrix(signals)
    assert mat[0, 0] == 0.0
    assert mat[1, 1] == 0.0
    assert np.all(mat[0, :] == 0.0)
    assert np.all(mat[:, 0] == 0.0)
    assert np.all(mat[1, :] == 0.0)
    assert np.all(mat[:, 1] == 0.0)

    # Valid regions still have correlation 1.0
    assert abs(mat[2, 3] - 1.0) < 1e-4

    signal_dict = {"signals": signals, "fps": fps, "bpm": 72.0}
    feats = coherence_features(signal_dict)
    assert np.isfinite(feats["coherence_mean"])
    assert feats["coherence_mean"] > 0.9
