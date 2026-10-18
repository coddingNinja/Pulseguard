# PulseGuard: Task Evaluation & Team Plan

Team: Ahmad Haseeb Butt (**A**), Muhammad Hamza Tahir (**B**), Hafiz Hamza Ahmad (**C**)
Estimates are focused work hours, not calendar time. Core scope = 117 h, 39 h per person.

---

## 1. Evaluation of current tasks

Checked on: clarity, feasibility, estimate, dependencies, acceptance criteria (AC).

| Task (from comp.md) | Issue | Fix |
|---|---|---|
| M1 (Days 1-3): rPPG extraction + validation | Four jobs hidden in one line (tracking, ROI, CHROM/POS, validation). No AC for "verified". No hour estimate. | Split into A1-A5. AC: BPM MAE ≤ 5 on UBFC-rPPG. |
| M2 (Days 4-7): Feature pipeline on Celeb-DF / FF++ | **Biggest risk.** Both datasets need an access request form, and approval can take days. FF++ raw is very large. Celeb-DF v2 is already compressed, so it is not "raw or low-compression". Depends on M1, but starts only after it. | Start access requests on Day 1 (B2). Cache ROI traces so features can be recomputed without re-decoding. Fix subset size (e.g. 200 real + 200 fake). Report results per compression level. |
| M3 (Days 8-11): Training, evaluation, error analysis | No metric or target. No rule for train/test split. Same-identity leakage would inflate results. "Fixes" is vague. | Split by identity/source video. Target: AUC reported with 95% CI; cross-dataset test required. Error analysis = 20 failure cases, categorised. |
| M4 (Days 12-14): Demo + presentation | Starts too late. UI and slides are blocked until Day 12, so any pipeline bug becomes a demo-day bug. | UI skeleton with mock data from Day 2 (C1). Integration at Day 11. Backup video on Day 13. |
| F1 rPPG extraction engine | Unclear what it returns. No interface between modules. | Freeze a function contract on Day 1 (see section 4). |
| F2 Feature + classifier module | Depends on F1 but no feature list is frozen. No baseline to beat. | Feature list fixed (peak, SNR, BPM, coherence, per-region variance). Baseline = SNR threshold, then XGBoost/SVM. |
| F3 Explainability views | No AC. Heatmap needs per-region data that F1 must expose. | C builds against mock data first. AC: shows waveform and 16-region heatmap for any clip. |
| F4 Live webcam mode | Stable BPM needs 10-15 s of video, so "real-time" is misleading. No fallback if lighting is bad. | AC: BPM within 10 of a pulse oximeter/phone reading after a 15 s warm-up. Show quality indicator. |
| F5 Video upload mode | "Confidence score" is undefined. A compressed real video also has no pulse, so "real/fake" overclaims. | Output a "pulse-consistency score" with a quality flag and an "unreliable" state. |
| F6 Evaluation report | No structure, no owner. Left to the last days. | Template created early (C5). Per-dataset table, ROC, failure gallery, limitations. |
| F7 Ensemble (stretch) | Needs pretrained weights and extra data. Easy to start and not finish. | Hard cut line: starts only if B6 finishes by Day 11. Otherwise listed as future work. |
| CNN (optional) | Same as F7. 12+ h with no safety net. | Cut. Keep as future work. |
| Division of work | Left blank. | Filled in sections 2-3. |
| Success metric | None in the document. | AUC target set as a goal, not a promise. A weak result with honest analysis is acceptable. |

---

## 2. Rewritten tasks (with acceptance criteria)

| ID | Task | Est | Acceptance criteria |
|---|---|---|---|
| A1 | Face tracking + landmark stabilisation (MediaPipe Face Mesh) | 6 h | Tracks face in ≥ 95% of frames on 5 test clips; jitter smoothed; handles frames with no face. |
| A2 | ROI extraction: 16 regions, mean RGB per frame | 6 h | Returns array `[16, T, 3]`; ROI masks drawn on a sample frame; saved to disk cache. |
| A3 | rPPG recovery: CHROM + POS, 0.7-4 Hz band-pass | 8 h | Clean waveform on webcam clip; both methods switchable by flag. |
| A4 | Spectral peak, SNR, BPM estimation | 5 h | Returns BPM, SNR, peak power per region and overall; unit tests on a synthetic sine signal. |
| A5 | Validate on UBFC-rPPG | 6 h | MAE ≤ 5 BPM vs ground truth on ≥ 10 subjects; error table saved. |
| A6 | Cross-region coherence features | 5 h | Pairwise correlation matrix + coherence score per clip; documented in README. |
| A7 | Backup demo video (screen recording) | 3 h | 2-3 min video covering live pulse, upload, real vs fake result. |
| B1 | Repo, environment, interface contract | 3 h | `requirements.txt`, folder layout, contract in section 4 agreed by all. |
| B2 | Dataset access, download, subset, splits | 6 h | Requests sent Day 1; subset of ≥ 200 real + 200 fake; `splits.csv` split by identity; compression level recorded per clip. |
| B3 | Batch extraction runner (parallel, cached, resumable) | 8 h | Runs the whole subset unattended; skips finished clips; failure log for clips with no face. |
| B4 | Baseline classifier (SNR rule, then XGBoost/SVM) | 8 h | Trained on train split only; AUC, accuracy, F1 on held-out test; saved model file. |
| B5 | Cross-dataset + per-quality evaluation | 6 h | Train on one dataset, test on the other; table by compression level; ROC plot. |
| B6 | Error analysis + fixes | 8 h | 20 failure cases categorised (compression, motion, makeup, pose); at least 2 fixes tried with before/after numbers. |
| C1 | Waveform plot + region heatmap (built on mock data first) | 6 h | Functions `plot_waveform()` and `plot_heatmap()` work on mock data, then on real A-output. |
| C2 | Live webcam mode | 8 h | Live BPM after 15 s warm-up; quality indicator; works on 2 different laptops. |
| C3 | Video upload mode + score | 5 h | Upload mp4, get score + quality flag + "unreliable" state in under 60 s for a 10 s clip. |
| C4 | Streamlit app integration | 6 h | Single `app.py`; both modes working; no crashes on 10 test clips. |
| C5 | Evaluation report | 6 h | Per-dataset table, ROC, failure gallery, limitations section. |
| C6 | Slides + rehearsal | 8 h | 5-7 minute deck; 2 full rehearsals; Q&A list of 10 likely questions. |
| S1 | *(Stretch)* EfficientNet ensemble | 10 h | Only if B6 is done by Day 11. Otherwise moved to future work. |

---

## 3. Work split (balanced, parallel)

### Ahmad (A): rPPG engine, 39 h
Order: **A1 → A2 → A3 → A4 → A5 → A6 → A7**
- Days 1-3: A1-A3. A3 v0 (even a rough version) must be ready by **Day 3**, because B3 and C2 need it.
- Days 4-6: A4, A5, A6.
- Day 13: A7.
- Depends on: B1 (contract). A5 needs the UBFC download (A downloads it himself, it is small).
- Blocks: B3 (needs A3), B4 (needs A4, A6), C2 (needs A3, A4).

### Muhammad Hamza Tahir (B): data + model, 39 h
Order: **B1 → B2 → B3 → B4 → B5 → B6**
- Day 1: B1 (finish same day), send dataset access requests.
- Days 1-4: B2 (while waiting for approvals, test with any available sample clips).
- Days 4-7: B3.
- Days 7-9: B4. Days 9-10: B5. Days 10-11: B6.
- Depends on: B2 → B3 (data), A3 → B3 (pipeline), B3 + A4 + A6 → B4, B4 → B5 → B6.
- Blocks: C3 (needs B4 model), C5 (needs B5, B6).

### Hafiz Hamza Ahmad (C): demo, report, slides, 39 h
Order: **C1 → C2 → C3 → C4 → C5 → C6**
- Days 2-5: C1 (mock data only, so no blocker).
- Days 5-8: C2 (needs A3 v0 and A4).
- Days 9-11: C3 (needs B4 model).
- Days 11-12: C4 (integration).
- Days 11-13: C5 (needs B5, B6 numbers). Days 12-14: C6.
- Depends on: B1 (contract), A3/A4 → C2, B4 → C3, B5/B6 → C5.

### Dependency summary

```
B1 ─┬─> A1 → A2 → A3 → A4 → A5
    │                 └──> A6
    ├─> B2 ──┐
    │   A3 ──┴─> B3 → B4 → B5 → B6
    └─> C1 ─> C2 (needs A3, A4)
B4 ─> C3 ─> C4 ─> C6
B5, B6 ─> C5 ─> C6
A4, A6, B4 ─> (A7 backup video after C4)
```

### Sync points
- **Day 1:** contract frozen, dataset requests sent.
- **Day 3:** A3 v0 working. If late, B3 uses saved ROI traces from webcam clips.
- **Day 7:** first full features on the subset. Go/no-go check on whether the signal separates real from fake.
- **Day 11:** model + app integrated. Stretch decision (S1 yes/no).
- **Day 13:** feature freeze. Only bug fixes after this.

---

## 4. Interface contract (agree on Day 1)

```python
extract_rois(video_path) -> np.ndarray      # [16, T, 3] mean RGB per region per frame, plus fps
recover_pulse(rois, fps, method="pos")      # -> dict(signals[16, T], bpm, snr, peak_power)
compute_features(signal_dict) -> np.ndarray # fixed-order feature vector
predict(video_path) -> dict(score, bpm, quality_flag, signals, region_coherence)
```

## 5. Risks to watch

| Risk | Early warning | Plan B |
|---|---|---|
| Dataset approval is slow | No reply by Day 3 | Use whatever is already accessible (e.g. publicly shared subsets) + record own real/fake clips |
| Features do not separate classes at Day 7 | AUC near 0.5 on a first test | Switch to FF++ raw only; report per-quality results; frame the result as honest limitation analysis |
| Compression kills the pulse | SNR similar for real and fake | Report by compression level; demo only on high-quality clips |
| Live demo fails on stage | Any crash in rehearsal | Backup video (A7) |
