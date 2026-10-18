"""A4 / A6: Spectral features, signal quality, and cross-region coherence.

Computes a deterministic, fixed-order 1D feature vector from recover_pulse() output.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.signal import welch

from .coherence import coherence_features
from .pulse import BP_HIGH, BP_LOW
from .roi import N_REGIONS, REGION_NAMES

# Fixed-order feature names list
FEATURE_NAMES: List[str] = [
    # Overall spectral features (4)
    "overall_bpm",
    "overall_snr",
    "overall_peak_power",
    "overall_spectral_peak_ratio",
    # Coherence & consistency features (4)
    "coherence_mean",
    "coherence_std",
    "coherence_min",
    "bpm_agreement_ratio",
]

# Per-region features (16 * 4 = 64)
for name in REGION_NAMES:
    FEATURE_NAMES.append(f"{name}_bpm")
    FEATURE_NAMES.append(f"{name}_snr")
    FEATURE_NAMES.append(f"{name}_peak_power")
    FEATURE_NAMES.append(f"{name}_variance")

TOTAL_FEATURES = len(FEATURE_NAMES)  # 4 + 4 + 64 = 72


def _region_spectral_stats(signal: np.ndarray, fps: float) -> tuple[float, float, float]:
    """Calculate bpm, snr (dB), and peak_power for one region signal."""
    if not np.isfinite(signal).all():
        return 0.0, -20.0, 0.0
    std_val = float(np.std(signal))
    if std_val < 1e-8 or fps <= 8.5:
        return 0.0, -20.0, 0.0

    T = len(signal)
    nperseg = min(T, int(round(fps * 8)))
    if nperseg < 4:
        return 0.0, -20.0, 0.0

    nfft = max(2048, nperseg)
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, nfft=nfft)

    in_band = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
    if not np.any(in_band):
        return 0.0, -20.0, 0.0

    f_band = freqs[in_band]
    psd_band = psd[in_band]

    max_idx = np.argmax(psd_band)
    peak_f = float(f_band[max_idx])
    peak_power = float(psd_band[max_idx])
    bpm = peak_f * 60.0

    harmonic_f = 2.0 * peak_f
    sig_mask = (np.abs(f_band - peak_f) <= 0.1) | (np.abs(f_band - harmonic_f) <= 0.1)
    p_sig = np.sum(psd_band[sig_mask])
    p_noise = np.sum(psd_band[~sig_mask])

    if p_noise > 1e-12 and p_sig > 1e-12:
        snr = float(10.0 * np.log10(p_sig / p_noise))
    elif p_sig > 1e-12:
        snr = 50.0
    else:
        snr = -20.0

    return bpm, snr, peak_power


def _compute_spectral_peak_ratio(signal: np.ndarray, fps: float) -> float:
    """Ratio of power at peak frequency to total in-band spectral power."""
    if not np.isfinite(signal).all() or len(signal) < 4 or fps <= 8.5:
        return 0.0
    T = len(signal)
    nperseg = min(T, int(round(fps * 8)))
    if nperseg < 4:
        return 0.0
    nfft = max(2048, nperseg)
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, nfft=nfft)
    in_band = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
    if not np.any(in_band):
        return 0.0
    psd_band = psd[in_band]
    total_power = np.sum(psd_band)
    if total_power < 1e-12:
        return 0.0
    return float(np.max(psd_band) / total_power)


def feature_dict(signal_dict: dict) -> Dict[str, float]:
    """Turn the output of recover_pulse into a dictionary of named features."""
    signals = signal_dict.get("signals")
    fps = float(signal_dict.get("fps", 30.0))
    overall = signal_dict.get("overall")

    # Overall features
    overall_bpm = float(signal_dict.get("bpm", np.nan))
    overall_bpm = overall_bpm if np.isfinite(overall_bpm) else 0.0

    overall_snr = float(signal_dict.get("snr", np.nan))
    overall_snr = overall_snr if np.isfinite(overall_snr) else -20.0

    overall_peak_power = float(signal_dict.get("peak_power", np.nan))
    overall_peak_power = overall_peak_power if np.isfinite(overall_peak_power) else 0.0

    peak_ratio = _compute_spectral_peak_ratio(overall, fps) if overall is not None else 0.0

    # Coherence features (Task A6)
    coh = coherence_features(signal_dict)

    f_dict: Dict[str, float] = {
        "overall_bpm": overall_bpm,
        "overall_snr": overall_snr,
        "overall_peak_power": overall_peak_power,
        "overall_spectral_peak_ratio": peak_ratio,
        "coherence_mean": coh["coherence_mean"],
        "coherence_std": coh["coherence_std"],
        "coherence_min": coh["coherence_min"],
        "bpm_agreement_ratio": coh["bpm_agreement_ratio"],
    }

    # Per-region features
    for r, name in enumerate(REGION_NAMES):
        if signals is not None and r < len(signals) and np.isfinite(signals[r]).all():
            sig = signals[r]
            r_bpm, r_snr, r_pwr = _region_spectral_stats(sig, fps)
            var_val = float(np.var(sig))
        else:
            r_bpm, r_snr, r_pwr, var_val = 0.0, -20.0, 0.0, 0.0

        f_dict[f"{name}_bpm"] = r_bpm
        f_dict[f"{name}_snr"] = r_snr
        f_dict[f"{name}_peak_power"] = r_pwr
        f_dict[f"{name}_variance"] = var_val

    return f_dict


def compute_features(signal_dict: dict) -> np.ndarray:
    """Turn the output of recover_pulse into a fixed-order 1D feature vector.

    Args:
        signal_dict: dictionary returned by recover_pulse().

    Returns:
        np.ndarray [72]: fixed-order 1D float64 feature vector.
    """
    f_dict = feature_dict(signal_dict)
    vec = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for i, name in enumerate(FEATURE_NAMES):
        vec[i] = f_dict[name]
    return vec
