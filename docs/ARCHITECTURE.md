# PulseGuard System Architecture

## 1. Pipeline Data Flow

```text
               Video Feed (Webcam or File)
                          │
                          ▼
            [ A1: MediaPipe FaceMesh ]  (pulseguard/tracking.py)
            - 468 landmark coordinates
            - Exponential smoothing (alpha=0.6)
            - 10-frame dropout hold
                          │
                          ▼
            [ A2: 16-Region Extraction ]  (pulseguard/roi.py)
            - 16 symmetric skin regions
            - Mean RGB spatial averaging
            - Linear interpolation for missing frames
                          │
                          ▼
                  rois [16, T, 3]  (Cached in data/cache/*.npz)
                          │
                          ▼
            [ A3: Pulse Recovery Engine ]  (pulseguard/pulse.py)
            - POS (Wang et al. 2017) & CHROM (de Haan 2013)
            - 3rd-order Butterworth bandpass (0.7 - 4.0 Hz)
            - Welch Power Spectral Density (BPM, SNR, Peak Power)
                          │
                          ▼
                  signal_dict (signals [16, T], overall [T])
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
    [ A6: Spatial Coherence ]  [ A4: Feature Vector ]
    (pulseguard/coherence.py) (pulseguard/features.py)
    - 16x16 correlation matrix - 72-element fixed 1D vector
    - Coherence mean/std/min   - Overall + Regional stats
              │                       │
              └───────────┬───────────┘
                          ▼
           [ B4: Deepfake Classifier ]  (pulseguard/classifier.py)
           - XGBoost / SVM Trained Model
           - Threshold SNR Rules
                          │
                          ▼
           [ Output & Explainability Views ]  (pulseguard/viz.py, app.py)
           - Real/Fake Probability (0.0 to 1.0)
           - Heart Rate & Quality Assessment (good / noisy / unreliable)
           - Pulse Waveform & 16-Region Coherence Heatmap
```

---

## 2. Module Ownership & Team Responsibilities

| Module | Primary Owner | Description | Status |
|---|---|---|---|
| `pulseguard/tracking.py` | Ahmad Haseeb Butt (A) | Face tracking & landmark smoothing (A1) | Done |
| `pulseguard/roi.py` | Ahmad Haseeb Butt (A) | 16-region ROI extraction & caching (A2) | Done |
| `pulseguard/pulse.py` | Ahmad Haseeb Butt (A) | POS & CHROM pulse recovery + bandpass (A3) | Done |
| `pulseguard/features.py` | Ahmad Haseeb Butt (A) | 72 fixed-order feature extraction (A4) | Done |
| `pulseguard/coherence.py` | Ahmad Haseeb Butt (A) | 16x16 coherence matrix & agreement ratio (A6) | Done |
| `pulseguard/pipeline.py` | Ahmad Haseeb Butt (A) | End-to-end glue & quality assessment | Done |
| `pulseguard/classifier.py` | M. Hamza Tahir (B) | Baseline rules & XGBoost/SVM model (B4) | Stub (Scaffolded) |
| `pulseguard/viz.py` | Hafiz Hamza Ahmad (C) | Waveform & 16-region coherence heatmap (C1) | Ready |
| `app.py` | Hafiz Hamza Ahmad (C) | Streamlit interactive web application (C2-C4) | Ready (Wired) |
| `docs/` | Team | Evaluation reports, guides, and presentation | In Progress |
