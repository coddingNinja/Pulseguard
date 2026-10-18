"""Tests for pulseguard.pulse (A3: pulse recovery)."""

import numpy as np
import pytest
from scipy.signal import welch

from pulseguard.pulse import recover_pulse


def generate_synthetic_rois(fps: float = 30.0, duration: float = 20.0,
                            target_hz: float = 1.2, seed: int = 42) -> np.ndarray:
    """Generate synthetic rois [16, T, 3] with pulse in green channel."""
    rng = np.random.default_rng(seed)
    T = int(round(fps * duration))
    t = np.arange(T) / fps

    rois = np.zeros((16, T, 3), dtype=np.float64)
    for r in range(16):
        # Baseline skin RGB values
        base_r = 130.0 + rng.uniform(-5.0, 5.0)
        base_g = 95.0 + rng.uniform(-5.0, 5.0)
        base_b = 80.0 + rng.uniform(-5.0, 5.0)

        # Slow drift (< 0.2 Hz)
        drift = 3.0 * np.sin(2 * np.pi * 0.08 * t + rng.uniform(0, 2 * np.pi))

        # Pulse signal (clean sinusoid in green channel, amplitude ~0.02 of baseline)
        phase = rng.uniform(0, 2 * np.pi)
        pulse = 1.5 * np.sin(2 * np.pi * target_hz * t + phase)

        # Random noise
        noise_r = rng.normal(0, 0.2, T)
        noise_g = rng.normal(0, 0.2, T)
        noise_b = rng.normal(0, 0.2, T)

        rois[r, :, 0] = base_r + drift + noise_r
        rois[r, :, 1] = base_g + drift + pulse + noise_g
        rois[r, :, 2] = base_b + drift + noise_b

    return rois


@pytest.mark.parametrize("method", ["pos", "chrom"])
def test_synthetic_72_bpm(method):
    fps = 30.0
    rois = generate_synthetic_rois(fps=fps, duration=20.0, target_hz=1.2)
    res = recover_pulse(rois, fps, method=method)

    assert res["method"] == method
    assert res["fps"] == fps
    assert res["signals"].shape == (16, rois.shape[1])
    assert res["overall"].shape == (rois.shape[1],)
    assert np.isfinite(res["bpm"])
    # Both methods must return bpm within +/-3 of 72
    assert abs(res["bpm"] - 72.0) <= 3.0, f"{method} returned bpm {res['bpm']} for target 72.0"
    assert res["snr"] > 0.0
    assert res["peak_power"] > 0.0


@pytest.mark.parametrize("target_hz, expected_bpm", [(0.9, 54.0), (2.0, 120.0)])
@pytest.mark.parametrize("method", ["pos", "chrom"])
def test_synthetic_various_frequencies(target_hz, expected_bpm, method):
    fps = 30.0
    rois = generate_synthetic_rois(fps=fps, duration=20.0, target_hz=target_hz)
    res = recover_pulse(rois, fps, method=method)

    assert abs(res["bpm"] - expected_bpm) <= 3.0, (
        f"{method} returned bpm {res['bpm']} for target {expected_bpm}"
    )


@pytest.mark.parametrize("method", ["pos", "chrom"])
def test_scaling_invariance(method):
    fps = 30.0
    rois = generate_synthetic_rois(fps=fps, duration=20.0, target_hz=1.2)
    rois_scaled = rois * 1.5

    res1 = recover_pulse(rois, fps, method=method)
    res2 = recover_pulse(rois_scaled, fps, method=method)

    s1 = res1["overall"]
    s2 = res2["overall"]

    corr = np.corrcoef(s1, s2)[0, 1]
    assert corr > 0.99, f"Correlation was {corr}"


@pytest.mark.parametrize("method", ["pos", "chrom"])
def test_band_limit(method):
    fps = 30.0
    rois = generate_synthetic_rois(fps=fps, duration=20.0, target_hz=1.2)
    res = recover_pulse(rois, fps, method=method)
    overall = res["overall"]

    freqs, psd = welch(overall, fs=fps, nperseg=int(fps * 8))
    in_band = (freqs >= 0.7) & (freqs <= 4.0)
    in_band_energy = np.sum(psd[in_band])
    total_energy = np.sum(psd)

    # Most of the output energy is inside 0.7-4 Hz
    assert in_band_energy / total_energy > 0.85


def test_nan_and_constant_region_handling():
    fps = 30.0
    rois = generate_synthetic_rois(fps=fps, duration=20.0, target_hz=1.2)
    # Set region 0 to NaNs, region 1 to constant
    rois[0, :, :] = np.nan
    rois[1, :, :] = 100.0

    res = recover_pulse(rois, fps, method="pos")
    assert np.isnan(res["signals"][0]).all()
    assert np.isnan(res["signals"][1]).all()
    # Other regions valid
    assert np.isfinite(res["signals"][2:]).all()
    # Overall is valid from other 14 regions
    assert np.isfinite(res["overall"]).all()
    assert abs(res["bpm"] - 72.0) <= 3.0


def test_invalid_method_raises():
    fps = 30.0
    rois = generate_synthetic_rois(fps=fps, duration=5.0)
    with pytest.raises(ValueError, match="Unknown pulse recovery method"):
        recover_pulse(rois, fps, method="fastica")


def test_invalid_fps_raises():
    rois = generate_synthetic_rois(fps=8.0, duration=5.0)
    with pytest.raises(ValueError, match="FPS must be > 8.5"):
        recover_pulse(rois, fps=8.0)


def test_too_short_input_does_not_crash():
    fps = 30.0
    # Input with very few frames (e.g. 5 frames, less than filtfilt padlen)
    rois = np.ones((16, 5, 3), dtype=np.float64) * 100.0
    res = recover_pulse(rois, fps=fps, method="pos")
    assert res["signals"].shape == (16, 5)
    assert np.isnan(res["signals"]).all()
    assert np.isnan(res["overall"]).all()
    assert np.isnan(res["bpm"])
    assert np.isnan(res["snr"])
