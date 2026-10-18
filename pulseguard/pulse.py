"""A3: rPPG recovery (POS / CHROM + band-pass filtering).

Turns per-region RGB traces [16, T, 3] into pulse waveforms [16, T], extracts
the combined pulse signal, and estimates heart rate (BPM), signal-to-noise
ratio (SNR), and spectral peak power.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, welch

MIN_FPS = 8.5
BP_LOW = 0.7   # Hz (42 BPM)
BP_HIGH = 4.0  # Hz (240 BPM)
BP_ORDER = 3


def _butter_bandpass(fps: float, order: int = BP_ORDER):
    nyq = 0.5 * fps
    low = BP_LOW / nyq
    high = BP_HIGH / nyq
    return butter(order, [low, high], btype="bandpass")


def _is_invalid_trace(C: np.ndarray) -> bool:
    """Return True if trace is all/partly NaN or constant."""
    if not np.isfinite(C).all():
        return True
    # If standard deviation is 0 in all channels, it is constant
    if np.all(np.std(C, axis=1) < 1e-8):
        return True
    return False


def _pos_channel(C: np.ndarray, fps: float, b, a) -> np.ndarray:
    """POS algorithm (Wang et al. 2017) on a single ROI trace C [3, T]."""
    T = C.shape[1]
    l = int(round(1.6 * fps))
    if l < 2 or T < l:
        return np.full(T, np.nan, dtype=np.float64)

    H = np.zeros(T, dtype=np.float64)
    # sliding window of length l, stride 1
    for n in range(T - l + 1):
        m = C[:, n : n + l]
        mean_m = np.mean(m, axis=1, keepdims=True)
        # Avoid divide by zero
        mean_m = np.where(np.abs(mean_m) < 1e-9, 1e-9, mean_m)
        Cn = m / mean_m
        # S = [[0, 1, -1], [-2, 1, 1]] @ Cn
        S = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64) @ Cn
        s1, s2 = S[0], S[1]
        std2 = np.std(s2)
        if std2 < 1e-9:
            alpha = 0.0
        else:
            alpha = float(np.std(s1) / std2)
        h = s1 + alpha * s2
        h = h - np.mean(h)
        H[n : n + l] += h

    # Padlen check for filtfilt
    padlen = 3 * max(len(a), len(b))
    if T <= padlen:
        return np.full(T, np.nan, dtype=np.float64)

    try:
        return filtfilt(b, a, H)
    except Exception:
        return np.full(T, np.nan, dtype=np.float64)


def _chrom_channel(C: np.ndarray, fps: float, b, a) -> np.ndarray:
    """CHROM algorithm (de Haan & Jeanne 2013) on a single ROI trace C [3, T]."""
    T = C.shape[1]
    win_len = int(round(1.6 * fps))
    padlen = 3 * max(len(a), len(b))
    if win_len <= padlen or T < win_len:
        return np.full(T, np.nan, dtype=np.float64)

    step = max(1, win_len // 2)
    hann = np.hanning(win_len)
    H_chrom = np.zeros(T, dtype=np.float64)
    weights = np.zeros(T, dtype=np.float64)

    for n in range(0, T - win_len + 1, step):
        m = C[:, n : n + win_len]
        mean_m = np.mean(m, axis=1, keepdims=True)
        mean_m = np.where(np.abs(mean_m) < 1e-9, 1e-9, mean_m)
        Cn = m / mean_m
        Rn, Gn, Bn = Cn[0], Cn[1], Cn[2]
        X = 3.0 * Rn - 2.0 * Gn
        Y = 1.5 * Rn + Gn - 1.5 * Bn
        try:
            Xf = filtfilt(b, a, X)
            Yf = filtfilt(b, a, Y)
        except Exception:
            continue
        std_yf = np.std(Yf)
        alpha = float(np.std(Xf) / std_yf) if std_yf > 1e-9 else 0.0
        S = Xf - alpha * Yf
        H_chrom[n : n + win_len] += S * hann
        weights[n : n + win_len] += hann

    idx = weights > 1e-6
    H_chrom[idx] /= weights[idx]
    if not np.any(idx):
        return np.full(T, np.nan, dtype=np.float64)

    # Fill unweighted ends if any before overall filtering
    if not np.all(idx):
        valid_indices = np.flatnonzero(idx)
        if len(valid_indices) < 2:
            return np.full(T, np.nan, dtype=np.float64)
        H_chrom = np.interp(np.arange(T), valid_indices, H_chrom[valid_indices])

    if T <= padlen:
        return np.full(T, np.nan, dtype=np.float64)

    try:
        return filtfilt(b, a, H_chrom)
    except Exception:
        return np.full(T, np.nan, dtype=np.float64)


def _spectral_analysis(signal: np.ndarray, fps: float) -> tuple[float, float, float]:
    """Calculate bpm, snr, and peak_power from Welch power spectral density.

    Returns:
        (bpm, snr, peak_power)
    """
    T = len(signal)
    if T < 2 or not np.isfinite(signal).all():
        return np.nan, np.nan, np.nan

    nperseg = min(T, int(round(fps * 8)))
    if nperseg < 4:
        return np.nan, np.nan, np.nan

    nfft = max(2048, nperseg)
    freqs, psd = welch(signal, fs=fps, nperseg=nperseg, nfft=nfft)

    in_band = (freqs >= BP_LOW) & (freqs <= BP_HIGH)
    if not np.any(in_band):
        return np.nan, np.nan, np.nan

    f_band = freqs[in_band]
    psd_band = psd[in_band]

    max_idx = np.argmax(psd_band)
    peak_f = float(f_band[max_idx])
    peak_power = float(psd_band[max_idx])
    bpm = peak_f * 60.0

    # SNR: power within +/-0.1 Hz of peak and 2nd harmonic / remaining power in 0.7-4.0 Hz
    harmonic_f = 2.0 * peak_f
    sig_mask = (np.abs(f_band - peak_f) <= 0.1) | (np.abs(f_band - harmonic_f) <= 0.1)

    p_sig = np.sum(psd_band[sig_mask])
    p_noise = np.sum(psd_band[~sig_mask])

    if p_noise > 1e-12 and p_sig > 1e-12:
        snr = float(10.0 * np.log10(p_sig / p_noise))
    elif p_sig > 1e-12:
        snr = 50.0
    else:
        snr = np.nan

    return bpm, snr, peak_power


def recover_pulse(rois: np.ndarray, fps: float, method: str = "pos") -> dict:
    """Turn per-region RGB traces into pulse signals.

    Args:
        rois: array [16, T, 3], mean RGB per region per frame (from extract_rois).
        fps: video frame rate (must be >= 8.5).
        method: "pos" or "chrom".

    Returns:
        dict with:
            "signals":    np.ndarray [16, T] band-passed pulse waveform per region
            "overall":    np.ndarray [T] combined waveform (mean of valid regions)
            "bpm":        float heart rate estimate in beats per minute
            "snr":        float signal-to-noise ratio in dB
            "peak_power": float power at spectral peak
            "method":     str ("pos" or "chrom")
            "fps":        float
    """
    method_lower = method.lower().strip()
    if method_lower not in ("pos", "chrom"):
        raise ValueError(f"Unknown pulse recovery method: {method!r}. Must be 'pos' or 'chrom'.")

    if not isinstance(fps, (int, float)) or fps <= MIN_FPS:
        raise ValueError(f"FPS must be > {MIN_FPS}, got {fps}")

    rois = np.asarray(rois, dtype=np.float64)
    if rois.ndim != 3 or rois.shape[0] != 16 or rois.shape[2] != 3:
        raise ValueError(f"Expected rois shape [16, T, 3], got {rois.shape}")

    n_regions, T, _ = rois.shape

    # Handle short clip
    b, a = _butter_bandpass(fps)
    padlen = 3 * max(len(a), len(b))
    if T <= padlen:
        signals = np.full((n_regions, T), np.nan, dtype=np.float64)
        overall = np.full(T, np.nan, dtype=np.float64)
        return {
            "signals": signals,
            "overall": overall,
            "bpm": np.nan,
            "snr": np.nan,
            "peak_power": np.nan,
            "method": method_lower,
            "fps": float(fps),
        }

    signals = np.full((n_regions, T), np.nan, dtype=np.float64)
    valid_signals = []

    for r in range(n_regions):
        # C is [3, T]
        C = rois[r].T
        if _is_invalid_trace(C):
            continue

        if method_lower == "pos":
            sig = _pos_channel(C, fps, b, a)
        else:
            sig = _chrom_channel(C, fps, b, a)

        signals[r] = sig
        if np.isfinite(sig).all():
            valid_signals.append(sig)

    if valid_signals:
        overall = np.mean(valid_signals, axis=0)
        bpm, snr, peak_power = _spectral_analysis(overall, fps)
    else:
        overall = np.full(T, np.nan, dtype=np.float64)
        bpm, snr, peak_power = np.nan, np.nan, np.nan

    return {
        "signals": signals,
        "overall": overall,
        "bpm": bpm,
        "snr": snr,
        "peak_power": peak_power,
        "method": method_lower,
        "fps": float(fps),
    }
