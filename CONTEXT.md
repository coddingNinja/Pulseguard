# PulseGuard — Project Status & Team Context

**Last Updated:** October 19, 2026  
**Project:** PulseGuard (Deepfake Detection Through Biological rPPG Signals)  
**Team Members:**
- **Ahmad Haseeb Butt (A):** rPPG Signal Engine (Tasks A1 – A7)
- **Muhammad Hamza Tahir (B):** Data Pipeline, Batch Extraction & Classification Models (Tasks B1 – B6)
- **Hafiz Hamza Ahmad (C):** Visualization, Explainability UI, Evaluation Report & Presentation (Tasks C1 – C6)

---

## 1. What We Have Done So Far (Completed & Verified)

All work completed so far corresponds to the core signal extraction engine (Ahmad's track A1 – A3):

### A1: Face Tracking & Landmark Stabilisation (`pulseguard/tracking.py`)
- Integrated MediaPipe's modern `FaceLandmarker` (Tasks API in VIDEO mode).
- Extracts 468 landmark coordinates (dropping the 10 iris points).
- Implemented `LandmarkSmoother` using exponential moving average (`alpha = 0.6`), jump reset protection, and gap-bridging for up to 10 dropped frames.

### A2: 16-Region ROI Extraction (`pulseguard/roi.py`)
- Defined 16 symmetric facial regions across forehead, temples, cheeks, chin, nose, and philtrum (`REGION_NAMES`).
- Implemented spatial mean RGB extraction per frame per region (`region_means`).
- Linear interpolation for missing frames (`interpolate_nans`).
- Disk caching mechanism (`.npz`) saving `rois [16, T, 3]`, `fps`, and `detection_rate`.
- Visual debugging overlay (`draw_rois`) with mask polygons and landmark numbering.

### A3: Pulse Recovery & Live Feedback (`pulseguard/pulse.py` & `pulseguard/live.py`)
- Implemented **POS** (Plane-Orthogonal-to-Skin, Wang et al. 2017) and **CHROM** (Chrominance-based, de Haan & Jeanne 2013).
- Implemented 3rd-order zero-phase Butterworth bandpass filter (0.7 to 4.0 Hz, corresponding to 42–240 BPM).
- Welch power spectral density estimation for Heart Rate (`bpm`), `snr` in dB ($\pm 0.1$ Hz around peak & 2nd harmonic), and `peak_power`.
- Live webcam integration in `pulseguard/live.py`: displays real-time 16 ROI overlays, rolling cheek green-channel intensity, rolling band-passed pulse waveform, and live estimated BPM after 10 s warm-up.
- Created `scripts/pulse_from_cache.py` to compare POS vs CHROM waveforms and frequency spectrum from `.npz` recordings.
- Created `tests/test_pulse.py`: verified scaling invariance, frequency accuracy (54, 72, 120 BPM), band-limiting, and error handling. Full test suite has **39/39 passing tests**.

### A4: Feature Vector Extraction (`pulseguard/features.py`)
- Implemented `compute_features(signal_dict)` returning a fixed-order 1D float64 vector of **72 features**.
- Overall spectral stats (bpm, snr, peak_power, spectral_peak_ratio).
- Spatial coherence features (mean, std, min off-diagonal correlation, and BPM agreement ratio).
- Per-region stats for each of the 16 facial regions (bpm, snr, peak_power, variance).
- Exported `FEATURE_NAMES` and inspection helper `feature_dict(signal_dict)`. Robust against NaNs and flat traces.

### A5: Ground-Truth UBFC-rPPG Validation (`scripts/validate_ubfc.py`)
- Automated validation runner comparing rPPG estimated BPM against contact sensor ground-truth.
- Verified mathematically via `--synthetic-selftest` (MAE = 0.23 BPM, RMSE = 0.26 BPM, Pearson r = 1.0000).
- Robust CLI handling: prompts with download instructions when `data/ubfc/` is missing.

### A6: Spatial Coherence Analysis (`pulseguard/coherence.py`)
- Implemented `coherence_matrix(signals)` computing 16x16 pairwise Pearson correlation matrix.
- Implemented `coherence_features(signal_dict)` computing mean/std/min correlation and fraction of regions agreeing within $\pm 5$ BPM of the overall heart rate.

### A7: Backup Demo Script (`docs/demo_script.md`)
- Prepared 2.5-minute shot-by-shot presentation script, pre-recording checklist, and live stage fallback contingency plans.

---

## 2. Shared Interface Contract (Frozen — Do Not Modify Signatures)

```python
# 1. ROI Extraction (pulseguard/roi.py)
extract_rois(video_path) -> (rois: np.ndarray [16, T, 3], fps: float)

# 2. Pulse Recovery (pulseguard/pulse.py)
recover_pulse(rois: np.ndarray, fps: float, method: str = "pos") -> dict(
    signals=np.ndarray [16, T],
    overall=np.ndarray [T],
    bpm=float,
    snr=float,
    peak_power=float,
    method=str,
    fps=float
)

# 3. Feature Computation (pulseguard/features.py)
compute_features(signal_dict: dict) -> np.ndarray  # Fixed-order 1D vector (72 features)

# 4. End-to-End Prediction (pulseguard/predict.py)
predict(video_path) -> dict(
    score=float,            # 0.0 (fake) to 1.0 (real)
    bpm=float,
    quality_flag=str,       # "good", "noisy", "unreliable"
    signals=np.ndarray [16, T],
    region_coherence=float,
    is_mock=bool
)
```

---

## 3. What Needs To Be Done Next (Task Breakdown by Person)

### Ahmad Haseeb Butt (A) — Status: A1–A4, A6, A7 COMPLETE:
- **A5:** Run benchmark on real UBFC dataset once downloaded into `data/ubfc/`.
- Record the 2–3 min backup demo video following `docs/demo_script.md`.

### Muhammad Hamza Tahir (B) — Next Steps:
- **B2:** Download and prepare datasets (Celeb-DF v2 / FF++ subsets, $\ge 200$ real + $\ge 200$ fake, split by identity in `data/splits.csv`).
- **B3:** Complete batch extraction implementation in `scripts/batch_extract.py`.
- **B4:** Train baseline classifier (SNR threshold rule, then XGBoost / SVM in `pulseguard/classifier.py` and `scripts/train.py`).
- **B5:** Cross-dataset & per-compression-level evaluation (`scripts/evaluate.py`).
- **B6:** Error analysis on 20 failure cases (compression, motion, pose, lighting).

### Hafiz Hamza Ahmad (C) — Next Steps:
- **C1:** Finalize any extra interactive options in `pulseguard/viz.py` (basic plots are working).
- **C2–C4:** Connect real model predictions into `app.py` once Task B4 model weights land in `models/classifier.pkl`.
- **C5 & C6:** Complete final evaluation report (`docs/report.md`), slides deck, and rehearsal.
