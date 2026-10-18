# PulseGuard Team Tasks & Execution Status

**Total Work Scope:** 117 focused hours (39 h / person).

---

## 1. Task Matrix

| ID | Task Description | Owner | Est. | Status | Acceptance Criteria |
|---|---|---|---|---|---|
| **A1** | Face tracking & landmark stabilisation | Ahmad (A) | 6 h | **DONE** | Tracks face in $\ge 95\%$ of frames; EMA smoothing; 10-frame gap holding. |
| **A2** | 16-region ROI extraction & caching | Ahmad (A) | 6 h | **DONE** | Returns `[16, T, 3]`; overlay check (`draw_rois`); disk cache `.npz`. |
| **A3** | rPPG recovery (POS / CHROM) + Live view | Ahmad (A) | 8 h | **DONE** | 0.7–4 Hz Butterworth; Welch PSD; live webcam plotting with 10s warm-up. |
| **A4** | Feature vector extraction (72 features) | Ahmad (A) | 5 h | **DONE** | `compute_features()` fixed-order 1D vector; handles NaNs/flat signals. |
| **A5** | UBFC-rPPG benchmark validation | Ahmad (A) | 6 h | **READY** | `scripts/validate_ubfc.py` completed; MAE $\le 5$ BPM (tested on synthetic). |
| **A6** | Cross-region spatial coherence | Ahmad (A) | 5 h | **DONE** | 16x16 correlation matrix; coherence score; BPM agreement ratio. |
| **A7** | Backup demo script & guidelines | Ahmad (A) | 3 h | **DONE** | `docs/demo_script.md` 2.5 min shot-by-shot script & contingency plan. |
| **B1** | Repo, environment, interface contract | Hamza Tahir (B) | 3 h | **DONE** | Frozen interface contract agreed in `docs/INTERFACE.md`. |
| **B2** | Dataset access, download, subset, splits | Hamza Tahir (B) | 6 h | Not Started | $\ge 200$ real + $\ge 200$ fake; split by identity in `data/splits.csv`. |
| **B3** | Batch extraction runner (parallel, cached) | Hamza Tahir (B) | 8 h | Scaffolded | `scripts/batch_extract.py` stub ready with CLI & multiprocessing skeleton. |
| **B4** | Baseline classifier (SNR, XGBoost / SVM) | Hamza Tahir (B) | 8 h | Scaffolded | `pulseguard/classifier.py` and `scripts/train.py` scaffolded. |
| **B5** | Cross-dataset & per-quality evaluation | Hamza Tahir (B) | 6 h | Scaffolded | `scripts/evaluate.py` stub with compression breakdown & ROC curve. |
| **B6** | Error analysis & fixes (20 cases) | Hamza Tahir (B) | 8 h | Not Started | Categorized failure cases (compression, motion, makeup, pose). |
| **C1** | Waveform plot + 16-region heatmap | Hamza Ahmad (C) | 6 h | **DONE** | `pulseguard/viz.py` working on real and mock data. |
| **C2** | Live webcam mode | Hamza Ahmad (C) | 8 h | **DONE** | Live BPM after warm-up; quality indicator; integrated in `app.py`. |
| **C3** | Video upload mode + confidence score | Hamza Ahmad (C) | 5 h | **DONE** | Upload tab in `app.py` wired to `predict()` and mock switcher. |
| **C4** | Streamlit app integration (`app.py`) | Hamza Ahmad (C) | 6 h | **DONE** | Multi-tab UI operational (`streamlit run app.py`). |
| **C5** | Evaluation report (`docs/report.md`) | Hamza Ahmad (C) | 6 h | In Progress | Per-dataset table, ROC curves, failure analysis. |
| **C6** | Slides deck & rehearsal | Hamza Ahmad (C) | 8 h | Not Started | 5–7 min deck, 2 full rehearsals, 10 likely Q&A. |
