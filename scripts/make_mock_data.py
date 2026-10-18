"""Generate realistic mock rPPG data (signal_dict and .npz cache) for testing without video.

Usage:
    python scripts/make_mock_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pulseguard.roi import REGION_NAMES
from pulseguard.utils import DEFAULT_CACHE_DIR, save_cache


def make_mock_signal_dict(is_real: bool = True, duration: float = 15.0, fps: float = 30.0, seed: int = 42) -> dict:
    """Generate a mock recover_pulse() output dictionary."""
    rng = np.random.default_rng(seed)
    T = int(round(duration * fps))
    t = np.arange(T) / fps
    n_regions = len(REGION_NAMES)

    signals = np.zeros((n_regions, T), dtype=np.float64)

    if is_real:
        bpm = 72.0
        target_hz = bpm / 60.0
        # Real face: all regions share common pulse with minor phase/amplitude variations
        for r in range(n_regions):
            phase = rng.uniform(-0.15, 0.15)
            amp = rng.uniform(0.8, 1.2)
            noise = rng.normal(0, 0.1, T)
            signals[r] = amp * np.sin(2 * np.pi * target_hz * t + phase) + noise
        overall = np.mean(signals, axis=0)
        snr = 8.5
        peak_power = 0.45
    else:
        bpm = 0.0
        # Synthetic deepfake: incoherent noise, inconsistent frequencies across regions
        for r in range(n_regions):
            freq = rng.uniform(0.5, 4.0)
            noise = rng.normal(0, 0.5, T)
            signals[r] = 0.3 * np.sin(2 * np.pi * freq * t) + noise
        overall = np.mean(signals, axis=0)
        snr = -4.2
        peak_power = 0.05

    return {
        "signals": signals,
        "overall": overall,
        "bpm": bpm if is_real else 64.0,
        "snr": snr,
        "peak_power": peak_power,
        "method": "pos",
        "fps": fps,
        "is_mock": True,
    }


def make_mock_cache():
    """Write mock .npz files for real-like and fake-like clips to data/cache/."""
    fps = 30.0
    T = 450
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Real-like ROIs: high pulse coherence in green channel
    rois_real = np.full((16, T, 3), 110.0, dtype=np.float64)
    t = np.arange(T) / fps
    for r in range(16):
        rois_real[r, :, 1] += 2.0 * np.sin(2 * np.pi * 1.2 * t)

    save_cache(
        DEFAULT_CACHE_DIR / "mock_real.npz",
        rois_real,
        fps,
        detection_rate=0.98,
        source="mock:real_face",
        region_names=REGION_NAMES,
    )

    # Fake-like ROIs: incoherent noise
    rng = np.random.default_rng(999)
    rois_fake = np.full((16, T, 3), 110.0, dtype=np.float64) + rng.normal(0, 1.0, (16, T, 3))
    save_cache(
        DEFAULT_CACHE_DIR / "mock_fake.npz",
        rois_fake,
        fps,
        detection_rate=0.95,
        source="mock:fake_face",
        region_names=REGION_NAMES,
    )

    print("Created mock caches:")
    print(f"  {DEFAULT_CACHE_DIR / 'mock_real.npz'}")
    print(f"  {DEFAULT_CACHE_DIR / 'mock_fake.npz'}")


if __name__ == "__main__":
    make_mock_cache()
