from pathlib import Path

import numpy as np
import pytest

from conftest import FakeTracker, make_video, fake_landmarks
from pulseguard.tracking import (DEFAULT_MODEL_PATH, N_LANDMARKS, FaceTracker,
                                 LandmarkSmoother, track_video)


# ---- smoother --------------------------------------------------------------
def test_first_detection_passes_through():
    s = LandmarkSmoother()
    lm = fake_landmarks()
    out, detected = s.update(lm)
    assert detected and np.allclose(out, lm)


def test_smoothing_reduces_jitter():
    rng = np.random.default_rng(1)
    base = fake_landmarks()
    s = LandmarkSmoother(alpha=0.5)
    raw_err, smooth_err = [], []
    for _ in range(100):
        noisy = base + rng.normal(0, 1.0, base.shape)
        out, _ = s.update(noisy)
        raw_err.append(np.mean((noisy - base) ** 2))
        smooth_err.append(np.mean((out - base) ** 2))
    assert np.mean(smooth_err[10:]) < 0.6 * np.mean(raw_err[10:])


def test_short_gap_is_carried_forward():
    s = LandmarkSmoother(max_gap=10)
    lm = fake_landmarks()
    s.update(lm)
    for _ in range(10):
        out, detected = s.update(None)
        assert out is not None and not detected
        assert np.allclose(out, lm)


def test_long_gap_becomes_invalid_and_recovers():
    s = LandmarkSmoother(max_gap=10)
    lm = fake_landmarks()
    s.update(lm)
    for _ in range(10):
        s.update(None)
    out, detected = s.update(None)       # 11th missing frame
    assert out is None and not detected
    out, detected = s.update(lm)         # face is back
    assert detected and np.allclose(out, lm)


def test_big_jump_resets_instead_of_smearing():
    s = LandmarkSmoother(alpha=0.2)
    lm = fake_landmarks()
    s.update(lm)
    moved = lm + np.array([60.0, 0.0])
    out, _ = s.update(moved)
    assert np.allclose(out, moved)


def test_bad_alpha_rejected():
    with pytest.raises(ValueError):
        LandmarkSmoother(alpha=0.0)


# ---- track_video -----------------------------------------------------------
def test_track_video_shapes_and_rate(tmp_path):
    video = make_video(tmp_path / "a.avi", n_frames=40)
    res = track_video(video, tracker=FakeTracker())
    assert res.landmarks.shape == (40, N_LANDMARKS, 2)
    assert res.detection_rate == 1.0
    assert res.valid.all() and res.detected.all()
    assert res.fps == pytest.approx(30, abs=0.5)


def test_no_face_video_does_not_crash(tmp_path):
    video = make_video(tmp_path / "blank.avi", n_frames=25)
    res = track_video(video, tracker=FakeTracker(all_missing=True))
    assert res.detection_rate == 0.0
    assert not res.valid.any()
    assert np.isnan(res.landmarks).all()


def test_gap_handling_in_video(tmp_path):
    video = make_video(tmp_path / "gap.avi", n_frames=60)
    res = track_video(video, tracker=FakeTracker(missing=range(20, 25)))  # 5-frame gap
    assert res.detection_rate == pytest.approx(55 / 60)
    assert res.valid.all()               # gap is shorter than max_gap -> carried forward

    res = track_video(video, tracker=FakeTracker(missing=range(20, 40)))  # 20-frame gap
    assert res.valid.sum() == 60 - 10    # only the first 10 missing frames are bridged


# ---- real MediaPipe (only runs when the model file is present) -------------
@pytest.mark.skipif(not Path(DEFAULT_MODEL_PATH).exists(),
                    reason="models/face_landmarker.task not downloaded")
def test_real_tracker_runs_on_blank_frame():
    with FaceTracker() as t:
        out = t.detect(np.zeros((120, 160, 3), np.uint8), 0)
    assert out is None  # no face in a black image
