"""Tests for pulseguard.features (A4: Feature extraction)."""

import numpy as np
import pytest

from pulseguard.features import FEATURE_NAMES, compute_features, feature_dict
from pulseguard.pulse import recover_pulse
from tests.test_pulse import generate_synthetic_rois


def test_features_fixed_length_and_names():
    fps = 30.0
    rois = generate_synthetic_rois(fps=fps, duration=15.0, target_hz=1.2)
    sig_dict = recover_pulse(rois, fps, method="pos")

    vec = compute_features(sig_dict)
    f_dict = feature_dict(sig_dict)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert len(vec) == len(FEATURE_NAMES)
    assert len(vec) == 72
    assert len(f_dict) == len(FEATURE_NAMES)

    # Check deterministic order
    for idx, name in enumerate(FEATURE_NAMES):
        assert vec[idx] == f_dict[name]


def test_features_synthetic_sine_known_bpm():
    fps = 30.0
    T = int(30.0 * 20.0)
    t = np.arange(T) / fps
    target_hz = 1.2  # 72 BPM

    # Coherent pulse across all 16 regions (as in real face)
    rois = np.zeros((16, T, 3), dtype=np.float64)
    for r in range(16):
        pulse = 1.5 * np.sin(2 * np.pi * target_hz * t)
        rois[r, :, 0] = 130.0
        rois[r, :, 1] = 95.0 + pulse
        rois[r, :, 2] = 80.0

    sig_dict = recover_pulse(rois, fps, method="pos")

    f_dict = feature_dict(sig_dict)
    assert abs(f_dict["overall_bpm"] - 72.0) <= 3.0
    assert f_dict["overall_snr"] > 0.0
    assert f_dict["overall_peak_power"] > 0.0
    assert f_dict["coherence_mean"] > 0.5
    assert f_dict["bpm_agreement_ratio"] > 0.5


def test_features_flat_and_nan_signals_robustness():
    fps = 30.0
    T = 300
    # Create invalid/flat signals
    signals = np.full((16, T), np.nan, dtype=np.float64)
    signals[0] = 0.0  # flat
    overall = np.full(T, np.nan, dtype=np.float64)

    bad_dict = {
        "signals": signals,
        "overall": overall,
        "bpm": np.nan,
        "snr": np.nan,
        "peak_power": np.nan,
        "fps": fps,
        "method": "pos",
    }

    vec = compute_features(bad_dict)
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 72
    assert np.isfinite(vec).all()  # Must not crash or return unhandled NaNs
