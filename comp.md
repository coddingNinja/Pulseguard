# PulseGuard: Deepfake Detection Through Biological Signals

> A prototype that detects synthetic face videos by checking for something real faces have and most deepfakes don't: a consistent, physiological blood-flow signal (rPPG).

**Team:** PulseGuard / Ahmad Haseeb Butt, Muhammad Hamza Tahir, Hafiz Hamza Ahmad
**Track:** AI / Trust & Safety

---

## 1. Project Overview & Problem Statement

Deepfakes are now cheap to make and hard to spot. Most detectors look for visual artifacts (blending seams, odd textures, unnatural blinking), and they break every time a new generator comes out. This creates a real trust problem for journalism, identity verification, video evidence, and social media.

Real faces carry a signal that generators don't explicitly model. With every heartbeat, blood volume under the skin changes slightly, which causes tiny color shifts in the face. This is called **remote photoplethysmography (rPPG)**. A real video has a consistent pulse signal across the face. A synthetic one usually has a weak, noisy, or spatially inconsistent signal.

**Problem:** Artifact-based detectors don't generalize to new generation methods. We need a complementary signal that is independent of how the fake was made.

## 2. Proposed Solution

PulseGuard extracts rPPG signals from different regions of a face in a video, measures how coherent they are, and uses a classifier to decide whether the video is likely real or synthetic.

**Pipeline**

1. **Face tracking:** MediaPipe Face Mesh follows the face and landmarks frame by frame.
2. **ROI extraction:** The face is split into 16-32 regions (forehead, cheeks, chin, etc.). We take the average color of each region per frame.
3. **rPPG signal recovery:** CHROM/POS methods turn color traces into pulse signals. A band-pass filter (0.7-4 Hz) isolates the heart-rate range.
4. **Feature computation:** Spectral peak, signal-to-noise ratio, estimated BPM, and cross-region coherence.
5. **Classification:** XGBoost/SVM on handcrafted features (baseline), with an optional small CNN on spatial-temporal PPG maps.
6. **Optional ensemble:** Combine the PPG score with a pretrained spatial detector (e.g., EfficientNet) to show that the two signals complement each other.
7. **Demo UI:** Upload a video or use a live webcam. See the pulse waveform, a region-consistency heatmap, and a real/fake score.

## 3. Objectives & Expected Outcomes

**Objectives**

- Build a working rPPG extraction pipeline and validate it against known ground truth.
- Show that PPG-based features separate real from synthetic videos on a benchmark subset.
- Deliver an interactive demo that makes the detection process visible and explainable.

**Expected outcomes**

- A functioning prototype with a measurable detection score on held-out data.
- Honest evaluation: per-dataset results, plus a clear picture of where the method fails.
- A live demo showing the user's own pulse extracted in real time.

We are not claiming parity with commercial systems. This is a research-grade prototype that demonstrates the approach.

## 4. Target Users

- **Journalists and fact-checkers** verifying suspicious video.
- **Identity verification / KYC platforms** that need liveness and authenticity checks.
- **Trust & safety teams** at media and social platforms.
- **Researchers and students** exploring physiological approaches to deepfake detection.

## 5. Key Features / Deliverables

| # | Feature | Description |
|---|---------|-------------|
| 1 | rPPG extraction engine | Face tracking, ROI signals, filtering, BPM estimation |
| 2 | Feature + classifier module | Handcrafted PPG features with XGBoost/SVM baseline |
| 3 | Explainability views | Pulse waveform plot and per-region consistency heatmap |
| 4 | Live webcam mode | Real-time pulse extraction for demo and sanity checks |
| 5 | Video upload mode | Analyze a clip and return a real/fake confidence score |
| 6 | Evaluation report | Metrics per dataset, failure case analysis |
| 7 | _(Stretch)_ Ensemble | PPG score combined with a spatial-artifact detector |

## 6. Implementation Plan & Timeline (14 days)

| Days | Milestone | Output |
|------|-----------|--------|
| 1-3 | rPPG extraction | Working pipeline; verified on webcam video and a validation dataset (UBFC-rPPG) |
| 4-7 | Feature pipeline | Features extracted on a subset of Celeb-DF v2 / FaceForensics++ (raw or low-compression) |
| 8-11 | Model training and evaluation | Trained classifier, metrics, error analysis, fixes |
| 12-14 | Demo and presentation | Streamlit/Gradio app, slides, rehearsal, backup demo video |

**Division of work:** _<assign owners: extraction / modeling / UI / presentation>_

## 7. Resources & Requirements

**Tech stack:** Python, OpenCV, MediaPipe, NumPy/SciPy, scikit-learn/XGBoost, PyTorch (for the optional CNN), Streamlit or Gradio.

**Datasets:**
- Celeb-DF v2 (subset) for real vs. fake evaluation.
- FaceForensics++ (raw/low-compression versions only; heavy compression destroys the PPG signal).
- UBFC-rPPG for validating pulse extraction.

**Hardware:** A laptop with a webcam. A GPU is helpful for the optional CNN but not required for the baseline.

## 8. Potential Challenges & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Video compression, low light, and low resolution degrade the pulse signal | Lower accuracy | Use raw or low-compression data; select demo clips carefully; report results by quality level |
| Weak cross-dataset generalization | Overfitting to one dataset | Evaluate on a second dataset; report numbers honestly |
| Newer deepfakes may preserve the source's pulse | False negatives | Document as a known limitation; ensemble with a spatial detector |
| Makeup, motion, and head pose add noise | False positives | Stabilize with landmark tracking; use longer clips (10 s or more) |
| Time constraint (2 weeks) | Scope creep | Feature-based baseline first; CNN and ensemble are stretch goals |

## 9. Additional Details

**Why this approach:** It targets a physical signal rather than generator-specific artifacts, which makes it a promising complement to existing detectors. It is also explainable: the judges and users can see the evidence (waveform, heatmap) rather than just a score.

**Background:** Based on the research direction of Ciftci, Demir, and Yin, "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals" (IEEE TPAMI), which Intel later productized.

**Future work:** Train the CNN on spatial-temporal PPG maps, adversarial testing against pulse-aware generators, real-time streaming support, and a browser extension or API for verification workflows.