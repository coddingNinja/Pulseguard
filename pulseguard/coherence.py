"""A6: Cross-region coherence and spatial consistency metrics.

Calculates pairwise correlation across the 16 facial region signals and
measures how coherently the pulse travels across the face.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .pulse import BP_HIGH, BP_LOW


def coherence_matrix(signals: np.ndarray) -> np.ndarray:
    """Compute the 16x16 pairwise Pearson correlation matrix across regions.

    Args:
        signals: array [16, T] of band-passed pulse signals per region.

    Returns:
        matrix: array [16, 16] with correlation coefficients in [-1.0, 1.0].
                If a region is NaN or constant, its row and column will be 0.0,
                with 1.0 on the diagonal for valid signals and 0.0 for invalid ones.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[0] != 16:
        raise ValueError(f"Expected signals shape [16, T], got {signals.shape}")

    n_regions, T = signals.shape
    matrix = np.zeros((n_regions, n_regions), dtype=np.float64)

    if T < 2:
        return matrix

    valid_mask = np.zeros(n_regions, dtype=bool)
    centered = np.zeros_like(signals)
    stds = np.zeros(n_regions, dtype=np.float64)

    for r in range(n_regions):
        sig = signals[r]
        if np.isfinite(sig).all():
            std_val = float(np.std(sig))
            if std_val > 1e-8:
                valid_mask[r] = True
                centered[r] = sig - np.mean(sig)
                stds[r] = std_val

    for i in range(n_regions):
        if not valid_mask[i]:
            continue
        matrix[i, i] = 1.0
        for j in range(i + 1, n_regions):
            if not valid_mask[j]:
                continue
            cov = float(np.mean(centered[i] * centered[j]))
            r_val = cov / (stds[i] * stds[j] + 1e-12)
            r_val = float(np.clip(r_val, -1.0, 1.0))
            matrix[i, j] = r_val
            matrix[j, i] = r_val

    return matrix


def _region_bpm(signal: np.ndarray, fps: float) -> float:
    """Compute peak BPM for an individual region signal."""
    if not np.isfinite(signal).all():
        return np.nan
    std_val = float(np.std(signal))
    if std_val < 1e-8 or fps <= 8.5:
        return np.nan
    T = len(signal)
    nperseg = min(T, int(round(fps * 8)))
    if nperseg < 4:
        return np.nan
    nfft = max(2048, nperseg)
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, nfft=nfft)
    mask = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
    if not np.any(mask):
        return np.nan
    peak_idx = np.argmax(psd[mask])
    peak_f = float(freqs[mask][peak_idx])
    return peak_f * 60.0


def coherence_features(signal_dict: dict) -> dict:
    """Compute scalar spatial coherence features from a signal_dict.

    Args:
        signal_dict: dict returned by recover_pulse() containing:
            - 'signals': np.ndarray [16, T]
            - 'bpm': float (overall estimated BPM)
            - 'fps': float

    Returns:
        dict with:
            - 'coherence_matrix': np.ndarray [16, 16]
            - 'coherence_mean': float (mean of upper triangle off-diagonal correlation)
            - 'coherence_std': float (std of off-diagonal correlation)
            - 'coherence_min': float (min off-diagonal correlation)
            - 'bpm_agreement_ratio': float (fraction of regions within 5 BPM of overall BPM)
    """
    signals = signal_dict["signals"]
    fps = float(signal_dict.get("fps", 30.0))
    overall_bpm = float(signal_dict.get("bpm", np.nan))

    mat = coherence_matrix(signals)
    n_regions = mat.shape[0]

    triu_idx = np.triu_indices(n_regions, k=1)
    off_diag = mat[triu_idx]

    # Mask out entries where either region was invalid (correlation is 0.0 and diagonal was 0.0)
    valid_regions = [bool(mat[i, i] == 1.0) for i in range(n_regions)]
    valid_pairs = []
    for idx_k, (i, j) in enumerate(zip(triu_idx[0], triu_idx[1])):
        if valid_regions[i] and valid_regions[j]:
            valid_pairs.append(off_diag[idx_k])

    if valid_pairs:
        v_arr = np.array(valid_pairs, dtype=np.float64)
        c_mean = float(np.mean(v_arr))
        c_std = float(np.std(v_arr))
        c_min = float(np.min(v_arr))
    else:
        c_mean = 0.0
        c_std = 0.0
        c_min = 0.0

    # Fraction of regions whose individual BPM agrees with overall BPM (within 5 BPM)
    agreements = 0
    counted = 0
    if np.isfinite(overall_bpm):
        for r in range(n_regions):
            if valid_regions[r]:
                r_bpm = _region_bpm(signals[r], fps)
                if np.isfinite(r_bpm):
                    counted += 1
                    if abs(r_bpm - overall_bpm) <= 5.0:
                        agreements += 1

    bpm_ratio = float(agreements / counted) if counted > 0 else 0.0

    return {
        "coherence_matrix": mat,
        "coherence_mean": c_mean,
        "coherence_std": c_std,
        "coherence_min": c_min,
        "bpm_agreement_ratio": bpm_ratio,
    }
