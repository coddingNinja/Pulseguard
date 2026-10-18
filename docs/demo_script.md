# A7: Backup Demo Script & Recording Guidelines

This guide specifies the 2–3 minute video recording script, pre-recording setup checklist, and contingency plans for demonstrating PulseGuard.

---

## 1. Pre-Recording Checklist

Before launching the demo or recording the screen:
- [ ] **Lighting:** Frontal, diffuse white light on the face. Avoid strong backlighting or flickering LED lamps.
- [ ] **Camera Positioning:** Sit ~50 cm from the webcam, eyes level with the top third of the frame.
- [ ] **Applications:** Close Zoom, FaceTime, Teams, and browser tabs using the camera to avoid device lock contention.
- [ ] **Warm-up:** Allow 10–15 seconds of remaining still for auto-exposure and bandpass filter initialization.
- [ ] **Environment:** Run within `.venv`:
  ```bash
  source .venv/bin/activate
  ```

---

## 2. Shot-by-Shot Demo Script (2.5 Minutes)

### Shot 1: Introduction & Problem Context (0:00 – 0:35)
- **Visual:** Title slide / `app.py` About tab.
- **Narrative:**
  > "Deepfakes are evolving rapidly, fooling traditional artifact-based detectors. PulseGuard takes a biological approach: genuine human faces exhibit microscopic, periodic color variations caused by blood pumped through facial capillaries (rPPG). Synthetic videos fail to reproduce this subtle, coherent biological signal."

### Shot 2: Live Webcam Biological Extraction (0:35 – 1:25)
- **Visual:** Terminal running `python scripts/webcam_roi.py --method pos` (or Streamlit Tab 2).
- **Actions:**
  1. Window opens showing 16 colored facial ROI masks tracked by MediaPipe.
  2. Point out the "FACE TRACKED" status and the scrolling raw green channel plot at the bottom.
  3. After the 10-second warm-up, highlight the orange band-passed pulse waveform appearing above and the live heart rate estimate (e.g., `BPM ~ 72 (est)`).
  4. Press `s` on the keyboard to demonstrate taking an instant annotated snapshot (`outputs/webcam_snapshot_*.png`).
  5. Press `q` to quit and show the saved `.npz` cache trace.

### Shot 3: Offline Analysis & Feature Explainability (1:25 – 1:55)
- **Visual:** Terminal running `python scripts/pulse_from_cache.py --latest`.
- **Actions:**
  1. Display the generated 3-panel comparison plot (`outputs/pulse_*.png`).
  2. Show POS vs. CHROM waveforms and the sharp cardiac fundamental frequency peak around 1.2 Hz (72 BPM) with SNR in dB.

### Shot 4: Deepfake Detection on Video Upload (1:55 – 2:30)
- **Visual:** Streamlit App (`streamlit run app.py` -> Video Upload tab).
- **Actions:**
  1. Upload a genuine test video -> Show high authenticity score (>90%), high coherence heatmap, and valid BPM.
  2. Upload a deepfake video -> Show low authenticity score (<20%), low coherence heatmap, and noisy frequency spectrum.
  3. Conclude with summary of defense robustness against generator visual artifacts.

---

## 3. Live Demo Fallback & Contingency Plan

If hardware, webcam permissions, or stage lighting fail during a live presentation:

1. **Fallback 1 (Video Source Mode):** Run the live UI on a pre-recorded test video:
   ```bash
   python scripts/webcam_roi.py --source data/raw/434_438.mp4
   ```
2. **Fallback 2 (Streamlit Mock Mode):** In `app.py`, check "Use Mock Pipeline Demo Data" to immediately demonstrate the UI, waveforms, and heatmaps without relying on real-time video decoding.
3. **Fallback 3 (Pre-recorded Demo Video):** Screen recording following this exact script kept on local desktop storage.
