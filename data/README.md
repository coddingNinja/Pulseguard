# PulseGuard Datasets & Splits Guide

This directory houses raw video datasets, extracted signal caches, and split definitions.

## 1. Directory Structure

- `data/raw/`: Store downloaded video files here (`.mp4`, `.avi`).
- `data/cache/`: Automatically holds `.npz` extracted ROI colour traces `[16, T, 3]`.
- `data/ubfc/`: Benchmark validation dataset (UBFC-rPPG).
- `data/splits.csv`: Master dataset split file mapping each video to train/val/test splits.

---

## 2. Dataset Setup & Acquisition

### A. Celeb-DF v2
- **Website:** [Celeb-DF on GitHub](https://github.com/yuezunli/celeb-deepfakeforensics)
- Submit the academic request form to obtain download links.
- Place sample clips into `data/raw/celeb_df/`.

### B. FaceForensics++ (FF++)
- **Website:** [FaceForensics on GitHub](https://github.com/ondyari/FaceForensics)
- Request access using the official form.
- Download `c23` (light compression) and `raw` subsets. Avoid `c40` (heavy compression removes subtle rPPG signals).
- Place sample clips into `data/raw/faceforensics/`.

### C. UBFC-rPPG (Pulse Validation Ground Truth)
- **Website:** [UBFC-rPPG Dataset](https://sites.google.com/view/ybvision/ubfc-rppg)
- Used for Task A5 to validate recovered pulse BPM against medical ground-truth sensor measurements.
- Structure inside `data/ubfc/`:
  ```text
  data/ubfc/
    subject1/
      vid.avi
      ground_truth.txt
    subject2/
      vid.avi
      ground_truth.txt
  ```

---

## 3. `splits.csv` Format

Your master splits file (`data/splits.csv`) must follow this schema (see [`data/splits_example.csv`](file:///Users/apple/Documents/pulseguard/data/splits_example.csv)):

| Column | Description | Example |
|---|---|---|
| `video_id` | Unique video identifier or filename without extension | `001_002` |
| `label` | Ground truth label (`real` or `fake`) | `fake` |
| `identity` | Person identifier (ensures zero identity leakage across splits) | `id_001` |
| `dataset` | Source benchmark dataset | `celeb_df_v2` |
| `compression_level` | Compression quality (`raw`, `c23`, `c40`) | `c23` |
| `split` | Assigned split (`train`, `val`, `test`) | `train` |
