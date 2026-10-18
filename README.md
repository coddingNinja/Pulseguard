# PulseGuard: Deepfake Detection Through Biological Signals

> A prototype detecting synthetic face videos by verifying genuine physiological blood-flow (**rPPG**) signals.

**Team:**
- **Ahmad Haseeb Butt:** rPPG extraction engine (Tracking, ROIs, POS/CHROM recovery, spectral & coherence features)
- **Muhammad Hamza Tahir:** Data splits, batch extraction, baseline & classifier modeling (XGBoost/SVM)
- **Hafiz Hamza Ahmad:** Explainability views, Streamlit application, evaluation report & slides

---

## 1. Quickstart

### Prerequisites & Installation
Python 3.10 – 3.13 on macOS / Linux / Windows.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the MediaPipe Face Mesh model file (`models/face_landmarker.task`):
```bash
python scripts/download_model.py
```

---

## 2. Running Tests

Run the full pytest suite (tracking, ROIs, pulse recovery, coherence, features, and contract validation):

```bash
pytest -q
```

---

## 3. Usage & Modes

### Live Webcam Pulse Extraction
Launch the real-time webcam monitor with POS pulse recovery and 10-second warm-up:
```bash
python scripts/webcam_roi.py --method pos
```
*(Options: `--method chrom`, `--mirror`, `--seconds 30`, `--camera 1`)*

### Interactive Streamlit Web Dashboard
Launch the multi-tab demo app (Video Upload Analysis, Live Webcam, and Methodology):
```bash
streamlit run app.py
```

### Offline Video Extraction & Pulse Analysis
1. Place your video in `data/raw/my_clip.mp4`.
2. Check facial masks on a sample frame:
   ```bash
   python scripts/draw_rois.py data/raw/my_clip.mp4
   ```
3. Extract 16 ROI traces to disk cache (`data/cache/my_clip.npz`):
   ```bash
   python scripts/run_roi_on_video.py data/raw/my_clip.mp4
   ```
4. Run POS & CHROM pulse analysis and generate comparison waveforms and spectra:
   ```bash
   python scripts/pulse_from_cache.py data/cache/my_clip.npz
   ```

### UBFC-rPPG Ground-Truth Validation
Verify heart rate recovery against contact sensor ground truth:
```bash
# Real dataset validation (once placed in data/ubfc/):
python scripts/validate_ubfc.py --data-dir data/ubfc --method pos

# Synthetic verification self-test:
python scripts/validate_ubfc.py --synthetic-selftest
```

### Generate Mock Data for UI/Modeling Work
Generate mock signal dictionaries and `.npz` cache files without video:
```bash
python scripts/make_mock_data.py
```

---

## 4. Documentation

- [System Architecture](file:///Users/apple/Documents/pulseguard/docs/ARCHITECTURE.md): Data flow diagram and module ownership map.
- [Frozen Interface Contract](file:///Users/apple/Documents/pulseguard/docs/INTERFACE.md): Exact signatures and worked example.
- [Team Task Matrix](file:///Users/apple/Documents/pulseguard/docs/TASKS.md): Task breakdown, hours, and status.
- [Feature Specification](file:///Users/apple/Documents/pulseguard/docs/features.md): 72-feature vector and coherence math.
- [UBFC Ground-Truth Validation](file:///Users/apple/Documents/pulseguard/docs/ubfc_validation.md): Ground-truth comparison and evaluation metrics.
- [Demo Recording Script](file:///Users/apple/Documents/pulseguard/docs/demo_script.md): 2.5-minute shot-by-shot presentation script.
- [Dataset Guide](file:///Users/apple/Documents/pulseguard/data/README.md): Setup instructions for Celeb-DF, FF++, and UBFC.
