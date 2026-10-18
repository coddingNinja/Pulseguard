# PulseGuard: rPPG engine (Ahmad's part: A1-A6)

Detects deepfakes by checking for a real heartbeat signal (rPPG) in face videos.
This repo holds the signal engine. **Done so far: A1 (face tracking) and A2 (ROI extraction).**
A3 (pulse recovery), A4/A6 (features) are stubs with the final signatures.

## Install

Python 3.10 - 3.13.

```bash
pip install -r requirements.txt
```

The first time the tracker runs it downloads MediaPipe's `face_landmarker.task`
(about 4 MB) into `models/`. To do it up front:

```bash
python scripts/download_model.py
```

If your network blocks the download, fetch the file manually from the URL printed
in the error and save it as `models/face_landmarker.task`.

## Try it on your own webcam clip

Record 15-20 s of your face (good light, sit still) and save it as `data/raw/me.mp4`.

```bash
python scripts/draw_rois.py data/raw/me.mp4          # check the 16 masks sit on skin
python scripts/draw_rois.py data/raw/me.mp4 --indices  # also shows landmark numbers, for tuning
python scripts/run_roi_on_video.py data/raw/me.mp4   # extract + print summary (cached)
python scripts/run_roi_on_video.py data/raw/me.mp4 --force   # ignore the cache
```

Always run `draw_rois.py` first. The region landmark lists in `pulseguard/roi.py`
are a first guess (see the TODO at the top of that file); fix them there if a
mask lands on an eye, lip, hair or the background. Only edit the right-hand lists;
the left side is mirrored automatically.

## Interface contract (do not change signatures)

```python
extract_rois(video_path) -> (rois, fps)       # rois: np.ndarray [16, T, 3], mean RGB per region per frame
recover_pulse(rois, fps, method="pos")        # -> dict(signals[16, T], bpm, snr, peak_power)   [A3, stub]
compute_features(signal_dict)                 # -> np.ndarray, fixed-order feature vector       [A4/A6, stub]
```

`extract_rois_ex(video_path)` is the same thing but also returns `detection_rate`.
Region order on axis 0 is `pulseguard.roi.REGION_NAMES` (16 names). "r"/"l" are the
subject's right/left.

Behaviour worth knowing:
- Frames where the face is lost are bridged by holding the last landmarks for up to
  10 frames. Longer gaps are filled by linear interpolation of the colour traces,
  so T always equals the number of video frames.
- A video with no face at all returns an all-NaN array and `detection_rate == 0`.
- Results are cached in `data/cache/<video_name>.npz` (rois, fps, detection_rate).
  Use `--force` / `force=True` after changing the smoother or the region lists.

## Layout

```
pulseguard/tracking.py   A1: MediaPipe FaceLandmarker + smoothing + gap handling
pulseguard/roi.py        A2: 16 regions, mean RGB, interpolation, cache, drawing
pulseguard/pulse.py      A3: stub
pulseguard/features.py   A4/A6: stub
pulseguard/utils.py      video reading, cache helpers
scripts/                 draw_rois.py, run_roi_on_video.py, download_model.py
tests/                   pytest (uses a fake tracker, so no model file needed)
```

## Tests

```bash
pytest
```

One test uses the real MediaPipe model and is skipped until `models/face_landmarker.task` exists.

## Note on MediaPipe versions

Recent MediaPipe releases (0.10.32 checked) no longer include the old
`mp.solutions.face_mesh` API, so this repo uses the newer `FaceLandmarker` (Tasks API).
It returns the same 468-point mesh; the 10 extra iris points are dropped.
