"""End-to-end pipeline execution and quality assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np

from .coherence import coherence_features
from .config import (
    MIN_DETECTION_RATE_GOOD,
    MIN_DETECTION_RATE_NOISY,
    MIN_SNR_GOOD,
    MIN_SNR_NOISY,
    QUALITY_GOOD,
    QUALITY_NOISY,
    QUALITY_UNRELIABLE,
)
from .features import compute_features
from .pulse import recover_pulse
from .roi import extract_rois_ex


def assess_quality(detection_rate: float, snr: float) -> str:
    """Classify video quality into 'good', 'noisy', or 'unreliable'.

    Args:
        detection_rate: fraction of frames with a face (0.0 to 1.0).
        snr: signal-to-noise ratio in dB.

    Returns:
        quality_flag in {"good", "noisy", "unreliable"}
    """
    if not np.isfinite(detection_rate) or detection_rate < MIN_DETECTION_RATE_NOISY:
        return QUALITY_UNRELIABLE
    if not np.isfinite(snr) or snr < MIN_SNR_NOISY:
        return QUALITY_UNRELIABLE

    if detection_rate >= MIN_DETECTION_RATE_GOOD and snr >= MIN_SNR_GOOD:
        return QUALITY_GOOD
    return QUALITY_NOISY


def run_pipeline(
    video_path: Union[str, Path],
    method: str = "pos",
    use_cache: bool = True,
    force: bool = False,
) -> Tuple[dict, np.ndarray, dict, str]:
    """Execute the full rPPG extraction, recovery, and feature computation pipeline.

    Args:
        video_path: path to video file.
        method: "pos" or "chrom".
        use_cache: whether to read/write cache from data/cache/.
        force: ignore existing cache.

    Returns:
        (signal_dict, feature_vector, coherence_dict, quality_flag)
    """
    res = extract_rois_ex(video_path, use_cache=use_cache, force=force)
    rois = res.rois
    fps = res.fps
    detection_rate = res.detection_rate

    signal_dict = recover_pulse(rois, fps, method=method)
    features = compute_features(signal_dict)
    coherence = coherence_features(signal_dict)
    quality_flag = assess_quality(detection_rate, signal_dict.get("snr", np.nan))

    return signal_dict, features, coherence, quality_flag
