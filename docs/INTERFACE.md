# Frozen Interface Contract

This document specifies the exact, immutable interface contract shared across all modules in PulseGuard.

---

## 1. Interface Signatures

```python
# 1. ROI Extraction (pulseguard/roi.py)
extract_rois(video_path: Union[str, Path]) -> Tuple[np.ndarray, float]
# Returns: (rois [16, T, 3], fps)

# 2. Pulse Recovery (pulseguard/pulse.py)
recover_pulse(rois: np.ndarray, fps: float, method: str = "pos") -> dict
# Returns dict with:
#   "signals":    np.ndarray [16, T] (band-passed pulse waveform per region)
#   "overall":    np.ndarray [T]     (combined waveform, mean of valid regions)
#   "bpm":        float              (heart rate from overall signal in 0.7-4.0 Hz)
#   "snr":        float              (signal-to-noise ratio in dB)
#   "peak_power": float              (power at the spectral peak)
#   "method":     str                ("pos" or "chrom")
#   "fps":        float

# 3. Feature Computation (pulseguard/features.py)
compute_features(signal_dict: dict) -> np.ndarray
# Returns: 1D np.ndarray [72] of float64 in deterministic FEATURE_NAMES order.

# 4. End-to-End Prediction (pulseguard/predict.py)
predict(video_path: Union[str, Path]) -> dict
# Returns dict with:
#   "score":            float (0.0=synthetic/fake, 1.0=real)
#   "bpm":              float
#   "quality_flag":     str in {"good", "noisy", "unreliable"}
#   "signals":          np.ndarray [16, T]
#   "region_coherence": float (mean off-diagonal correlation)
#   "is_mock":          bool (True until B4 model weights land)
```

---

## 2. Worked Example

```python
from pulseguard.roi import extract_rois
from pulseguard.pulse import recover_pulse
from pulseguard.features import compute_features, FEATURE_NAMES
from pulseguard.coherence import coherence_matrix
from pulseguard.predict import predict

# Step 1: Extract ROI color traces
rois, fps = extract_rois("data/raw/434_438.mp4")
print("ROIs shape:", rois.shape)  # e.g., (16, 337, 3)

# Step 2: Recover pulse
pulse_res = recover_pulse(rois, fps, method="pos")
print(f"BPM: {pulse_res['bpm']:.1f}, SNR: {pulse_res['snr']:.2f} dB")

# Step 3: Compute 72 features
features = compute_features(pulse_res)
print("Features vector shape:", features.shape)  # (72,)

# Step 4: End-to-end predict
prediction = predict("data/raw/434_438.mp4")
print(f"Authenticity score: {prediction['score']:.2f}, Quality: {prediction['quality_flag']}")
```
