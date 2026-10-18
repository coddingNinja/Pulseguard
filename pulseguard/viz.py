"""Owner: Hafiz Hamza Ahmad
Task ID: C1 / Explainability Views
Acceptance Criteria:
    - Functions plot_waveform() and plot_heatmap() work on mock data and real A-output.
    - Shows waveform and 16-region coherence heatmap for any clip.
Can import:
    matplotlib, numpy, plotly (optional)
TODO:
    1. Support optional interactive plotly outputs for Streamlit integration.
    2. Add color customisation and threshold indicators.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .roi import REGION_NAMES


def plot_waveform(signal_or_dict, fps: float = 30.0, title: str = "Recovered Pulse Waveform (rPPG)"):
    """Plot recovered rPPG waveform over time.

    Args:
        signal_or_dict: 1D np.ndarray [T] or signal_dict from recover_pulse().
        fps: frame rate.
        title: plot title.

    Returns:
        matplotlib.figure.Figure
    """
    if isinstance(signal_or_dict, dict):
        sig = signal_or_dict.get("overall", np.array([]))
        fps = float(signal_or_dict.get("fps", fps))
        bpm = signal_or_dict.get("bpm", np.nan)
        if np.isfinite(bpm):
            title = f"{title} — BPM ~ {bpm:.1f}"
    else:
        sig = np.asarray(signal_or_dict)

    T = len(sig)
    t = np.arange(T) / fps
    fig, ax = plt.subplots(figsize=(10, 3.2), constrained_layout=True)
    if T > 0 and np.isfinite(sig).any():
        ax.plot(t, sig, lw=1.3, color="#2563eb", label="Band-passed Pulse (0.7 - 4.0 Hz)")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No valid signal data", ha="center", va="center")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    return fig


def plot_heatmap(coherence_matrix: np.ndarray, region_names: list[str] = None):
    """Plot 16x16 correlation/coherence heatmap across facial regions.

    Args:
        coherence_matrix: 2D np.ndarray [16, 16] from coherence_matrix().
        region_names: optional list of 16 names (defaults to pulseguard.roi.REGION_NAMES).

    Returns:
        matplotlib.figure.Figure
    """
    if region_names is None:
        region_names = REGION_NAMES

    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    mat = np.asarray(coherence_matrix, dtype=np.float64)
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pairwise Pearson Correlation", rotation=270, labelpad=15)

    n = len(region_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(region_names, rotation=90, fontsize=8)
    ax.set_yticklabels(region_names, fontsize=8)
    ax.set_title("16-Region Pulse Coherence Heatmap", fontsize=11, fontweight="bold")
    return fig
