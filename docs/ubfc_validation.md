# A5: UBFC-rPPG Ground Truth Validation

## 1. Overview
Task A5 evaluates the physiological accuracy of PulseGuard's rPPG recovery engine (POS and CHROM methods) by comparing the estimated heart rate against reference contact PPG sensor data from the **UBFC-rPPG** dataset.

- **Target Acceptance Criteria:** Mean Absolute Error ($\text{MAE}$) $\le 5.0$ BPM on $\ge 10$ subjects.
- **Script:** [`scripts/validate_ubfc.py`](file:///Users/apple/Documents/pulseguard/scripts/validate_ubfc.py)

---

## 2. Dataset Acquisition

The UBFC-rPPG dataset is maintained by the Bourgogne Franche-Comté University vision laboratory:
- **Official Download:** [UBFC-rPPG Dataset Page](https://sites.google.com/view/ybvision/ubfc-rppg)
- **Directory Structure:**
  ```text
  data/ubfc/
    subject1/
      vid.avi              # Uncompressed video feed
      ground_truth.txt     # Reference PPG sensor waveform, HR, timestamps
    subject2/
      vid.avi
      ground_truth.txt
  ```

---

## 3. Real Dataset Benchmark Status

> [!NOTE]
> **Status:** Pending download of the raw UBFC-rPPG dataset into `data/ubfc/`.
> Once subject folders are placed in `data/ubfc/`, run:
> ```bash
> python scripts/validate_ubfc.py --data-dir data/ubfc --method pos
> ```
> Results will automatically save to `results/ubfc_errors.csv`.

---

## 4. Synthetic Verification Self-Test

To verify the mathematical evaluation logic (parsing, PSD peak extraction, error metric computation, and CSV reporting), the script includes a synthetic self-test:

```bash
python scripts/validate_ubfc.py --synthetic-selftest --method pos
```

### Self-Test Results Table (`results/ubfc_synthetic_selftest.csv`):

| Subject | Ground Truth (BPM) | Estimated (BPM) | Absolute Error (BPM) |
|---|---|---|---|
| `synth_subj_01` | 54.00 | 53.61 | 0.39 |
| `synth_subj_02` | 60.00 | 59.77 | 0.23 |
| `synth_subj_03` | 72.00 | 72.07 | 0.07 |
| `synth_subj_04` | 75.00 | 74.71 | 0.29 |
| `synth_subj_05` | 80.00 | 79.98 | 0.02 |
| `synth_subj_06` | 90.00 | 89.65 | 0.35 |
| `synth_subj_07` | 100.00 | 100.20 | 0.20 |
| `synth_subj_08` | 110.00 | 109.86 | 0.14 |
| `synth_subj_09` | 120.00 | 120.41 | 0.41 |
| `synth_subj_10` | 125.00 | 124.80 | 0.20 |

- **Mean Absolute Error (MAE):** 0.23 BPM
- **Root Mean Square Error (RMSE):** 0.26 BPM
- **Pearson correlation ($r$):** 1.0000
