"""Visualization utilities (C1): pulse waveform plots and 16-region coherence heatmaps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(signal: np.ndarray, fps: float, title: str = "Pulse Waveform"):
    """Plot recovered rPPG waveform over time."""
    T = len(signal)
    t = np.arange(T) / fps
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax.plot(t, signal, lw=1.2, color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    return fig


def plot_region_heatmap(coherence_matrix: np.ndarray, region_names: list[str]):
    """Plot 16x16 correlation/coherence heatmap across facial regions."""
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(coherence_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cross-Region Coherence")
    ax.set_xticks(range(len(region_names)))
    ax.set_yticks(range(len(region_names)))
    ax.set_xticklabels(region_names, rotation=90, fontsize=8)
    ax.set_yticklabels(region_names, fontsize=8)
    ax.set_title("16-Region Pulse Coherence Heatmap")
    return fig
