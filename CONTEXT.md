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

---

## 2. Shared Interface Contract (Frozen — Do Not Modify Signatures)

```python
# 1. ROI Extraction (Ahmad)
extract_rois(video_path) -> (rois: np.ndarray [16, T, 3], fps: float)

# 2. Pulse Recovery (Ahmad)
recover_pulse(rois: np.ndarray, fps: float, method: str = "pos") -> dict(
    signals=np.ndarray [16, T],
    overall=np.ndarray [T],
    bpm=float,
    snr=float,
    peak_power=float,
    method=str,
    fps=float
)

# 3. Feature Computation (Ahmad / Hamza Tahir)
compute_features(signal_dict: dict) -> np.ndarray  # Fixed-order 1D feature vector

# 4. End-to-End Prediction (Hamza Tahir / Hamza Ahmad)
predict(video_path) -> dict(
    score=float,            # 0.0 (fake) to 1.0 (real)
    bpm=float,
    quality_flag=str,       # "good", "noisy", "unreliable"
    signals=np.ndarray,
    region_coherence=float
)
```

---

## 3. What Needs To Be Done Next (Task Breakdown by Person)

### Ahmad Haseeb Butt (A) — Next Steps:
- **A4:** Finalize `pulseguard/features.py`: compute spectral peak, SNR, BPM, per-region variances.
- **A5:** Validate pulse recovery on benchmark ground-truth dataset (UBFC-rPPG, target: MAE $\le 5$ BPM).
- **A6:** Compute cross-region coherence features (16x16 correlation matrix & pairwise coherence score).
- **A7:** Record backup demo video showing live webcam pulse, video upload, and real vs. fake result.

### Muhammad Hamza Tahir (B) — Next Steps:
- **B2:** Prepare datasets & splits (Celeb-DF v2 / FaceForensics++ subset, $\ge 200$ real + $\ge 200$ fake, split by identity in `data/splits.csv`).
- **B3:** Build parallel batch extraction runner in `scripts/batch_extract.py` utilizing `extract_rois_ex` and `data/cache/`.
- **B4:** Train baseline classifier (SNR threshold rule, then XGBoost / SVM in `pulseguard/classifier.py` and `scripts/train.py`).
- **B5:** Cross-dataset & per-compression-level evaluation (`scripts/evaluate.py`).
- **B6:** Error analysis on 20 failure cases (compression, motion, pose, lighting).

### Hafiz Hamza Ahmad (C) — Next Steps:
- **C1:** Build visual components (`pulseguard/viz.py`): interactive pulse waveform plot and 16-region consistency heatmap.
- **C2 & C3:** Build the Streamlit demo application (`app.py`):
  - Tab 1: Live Webcam mode with live BPM and quality indicator.
  - Tab 2: Video upload mode with pulse-consistency score and "unreliable" quality flags.
- **C4:** App integration testing on real and fake sample clips.
- **C5 & C6:** Project evaluation report (`docs/report.md`), slides deck, and rehearsal.
