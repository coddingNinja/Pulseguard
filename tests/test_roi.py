import numpy as np
import pytest

from conftest import H, W, FakeTracker, fake_landmarks, make_video
from pulseguard.roi import (N_REGIONS, REGION_NAMES, REGIONS, draw_rois,
                            extract_rois, extract_rois_ex, interpolate_nans,
                            region_means)


# ---- region definitions ----------------------------------------------------
def test_sixteen_unique_regions():
    assert N_REGIONS == 16
    assert len(set(REGION_NAMES)) == 16


def test_left_regions_mirror_right_regions():
    from pulseguard.roi import _mirror
    for r, l in [("forehead_r_outer", "forehead_l_outer"),
                 ("forehead_r_inner", "forehead_l_inner"),
                 ("cheek_r_upper", "cheek_l_upper"),
                 ("cheek_r_lower", "cheek_l_lower"),
                 ("cheek_r_outer", "cheek_l_outer"),
                 ("chin_r", "chin_l")]:
        assert REGIONS[l] == _mirror(REGIONS[r])


# ---- colour averaging ------------------------------------------------------
def test_region_means_on_flat_frame():
    frame = np.empty((H, W, 3), np.uint8)
    frame[:] = (200, 120, 80)
    m = region_means(frame, fake_landmarks())
    assert m.shape == (16, 3)
    assert np.allclose(m, [200, 120, 80])


def test_region_means_nan_without_landmarks():
    frame = np.zeros((H, W, 3), np.uint8)
    assert np.isnan(region_means(frame, None)).all()


def test_region_means_with_face_partly_outside_frame():
    frame = np.full((H, W, 3), 100, np.uint8)
    lm = fake_landmarks() + np.array([70.0, 0.0])    # half the face off-screen
    m = region_means(frame, lm)                      # must not raise
    assert np.isfinite(m[np.isfinite(m)]).all()


def test_interpolate_nans():
    x = np.zeros((16, 10, 3))
    x[:] = np.arange(10)[None, :, None]
    broken = x.copy()
    broken[:, 3:6, :] = np.nan       # middle gap
    broken[:, 0, :] = np.nan         # leading gap
    fixed = interpolate_nans(broken)
    assert np.allclose(fixed[:, 3:6, :], x[:, 3:6, :])   # linear data is recovered exactly
    assert np.allclose(fixed[:, 0, :], 1.0)              # edge takes nearest valid value
    all_nan = np.full((16, 5, 3), np.nan)
    assert np.isnan(interpolate_nans(all_nan)).all()


# ---- extract_rois ----------------------------------------------------------
def test_output_shape_and_values(tmp_path):
    video = make_video(tmp_path / "v.avi", n_frames=40)
    rois, fps = extract_rois(video, cache_dir=tmp_path, tracker=FakeTracker())
    assert rois.shape == (16, 40, 3)
    assert fps == pytest.approx(30, abs=0.5)
    assert np.abs(rois - [200, 120, 80]).max() < 4       # MJPG is slightly lossy


def test_time_axis_follows_the_video(tmp_path):
    video = make_video(tmp_path / "ramp.avi", n_frames=40,
                       color_fn=lambda t: (100, 50 + 4 * t, 100))
    rois, _ = extract_rois(video, cache_dir=tmp_path, tracker=FakeTracker())
    g = rois[:, :, 1].mean(axis=0)
    assert g[-1] - g[0] > 100 and np.all(np.diff(g) > 0)


def test_cache_is_written_and_reused(tmp_path):
    video = make_video(tmp_path / "c.avi", n_frames=20)
    t1 = FakeTracker()
    r1 = extract_rois_ex(video, cache_dir=tmp_path, tracker=t1)
    assert not r1.from_cache and t1.calls == 20
    assert (tmp_path / "c.npz").exists()

    t2 = FakeTracker()
    r2 = extract_rois_ex(video, cache_dir=tmp_path, tracker=t2)
    assert r2.from_cache and t2.calls == 0               # tracker never touched
    assert np.array_equal(r1.rois, r2.rois) and r1.fps == r2.fps
    assert r2.detection_rate == r1.detection_rate

    t3 = FakeTracker()
    r3 = extract_rois_ex(video, cache_dir=tmp_path, tracker=t3, force=True)
    assert not r3.from_cache and t3.calls == 20


def test_use_cache_false_writes_nothing(tmp_path):
    video = make_video(tmp_path / "n.avi", n_frames=10)
    extract_rois(video, use_cache=False, cache_dir=tmp_path, tracker=FakeTracker())
    assert not (tmp_path / "n.npz").exists()


def test_cache_for_moved_video_is_not_reused(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    va = make_video(a / "same.avi", n_frames=10, color_fn=lambda t: (10, 10, 10))
    vb = make_video(b / "same.avi", n_frames=10, color_fn=lambda t: (240, 240, 240))
    extract_rois(va, cache_dir=tmp_path, tracker=FakeTracker())
    rois, _ = extract_rois(vb, cache_dir=tmp_path, tracker=FakeTracker())
    assert rois.mean() > 200                              # recomputed, not the dark clip


def test_no_face_video_returns_nan_not_crash(tmp_path):
    video = make_video(tmp_path / "blank.avi", n_frames=15)
    res = extract_rois_ex(video, cache_dir=tmp_path, tracker=FakeTracker(all_missing=True))
    assert res.rois.shape == (16, 15, 3)
    assert res.detection_rate == 0.0
    assert np.isnan(res.rois).all()


def test_long_gap_is_interpolated_to_keep_T(tmp_path):
    n = 60
    video = make_video(tmp_path / "gap.avi", n_frames=n,
                       color_fn=lambda t: (100, 50 + 2 * t, 100))
    res = extract_rois_ex(video, cache_dir=tmp_path,
                          tracker=FakeTracker(missing=range(20, 45)))   # 25 missing frames
    assert res.rois.shape == (16, n, 3)                   # T unchanged, nothing dropped
    assert np.isfinite(res.rois).all()
    assert res.detection_rate == pytest.approx((n - 25) / n)
    g = res.rois[:, :, 1].mean(axis=0)
    assert g[30] > g[19] and g[30] < g[45]                # interpolated between neighbours


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_rois(tmp_path / "nope.mp4", cache_dir=tmp_path, tracker=FakeTracker())


# ---- drawing ---------------------------------------------------------------
def test_draw_rois_returns_image_of_same_size():
    frame = np.full((H, W, 3), 128, np.uint8)
    out = draw_rois(frame, fake_landmarks())
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert not np.array_equal(out, frame)

    # Test toggling numbers and legend
    out_no_text = draw_rois(frame, fake_landmarks(), numbers=False, legend=False)
    assert out_no_text.shape == frame.shape and out_no_text.dtype == np.uint8
    assert not np.array_equal(out, out_no_text)
