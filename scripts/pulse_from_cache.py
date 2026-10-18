"""Load a cached webcam recording, run POS and CHROM pulse recovery, and plot results.

Usage:
    python scripts/pulse_from_cache.py data/cache/webcam_20261018_220000.npz
    python scripts/pulse_from_cache.py --latest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from pulseguard.pulse import recover_pulse
from pulseguard.utils import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, load_cache


def plot_pulse_comparison(cache_path: Path, out_path: Path):
    data = load_cache(cache_path)
    if data is None:
        sys.exit(f"Could not load cache file: {cache_path}")

    rois = data["rois"]
    fps = data["fps"]

    res_pos = recover_pulse(rois, fps, method="pos")
    res_chrom = recover_pulse(rois, fps, method="chrom")

    print(f"=== Pulse Recovery for {cache_path.name} ===")
    print(f"Measured FPS: {fps:.2f}")
    print(f"POS   -> BPM: {res_pos['bpm']:.1f}  | SNR: {res_pos['snr']:.2f} dB  | Peak Power: {res_pos['peak_power']:.4e}")
    print(f"CHROM -> BPM: {res_chrom['bpm']:.1f}  | SNR: {res_chrom['snr']:.2f} dB  | Peak Power: {res_chrom['peak_power']:.4e}")

    T = rois.shape[1]
    time_axis = np.arange(T) / fps

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    # 1. POS waveform
    axes[0].plot(time_axis, res_pos["overall"], color="tab:blue", lw=1.2, label=f"POS overall (BPM ~ {res_pos['bpm']:.1f})")
    axes[0].set_title(f"POS Pulse Waveform (band-passed 0.7-4.0 Hz)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # 2. CHROM waveform
    axes[1].plot(time_axis, res_chrom["overall"], color="tab:orange", lw=1.2, label=f"CHROM overall (BPM ~ {res_chrom['bpm']:.1f})")
    axes[1].set_title(f"CHROM Pulse Waveform (band-passed 0.7-4.0 Hz)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    # 3. Frequency spectra
    nperseg = min(T, int(round(fps * 8)))
    nfft = max(2048, nperseg)

    f_pos, psd_pos = welch(res_pos["overall"], fs=fps, nperseg=nperseg, nfft=nfft)
    f_chrom, psd_chrom = welch(res_chrom["overall"], fs=fps, nperseg=nperseg, nfft=nfft)

    # Filter to 0.5 - 4.5 Hz for display
    disp_mask = (f_pos >= 0.5) & (f_pos <= 4.5)
    axes[2].plot(f_pos[disp_mask], psd_pos[disp_mask] / (np.max(psd_pos[disp_mask]) + 1e-12),
                 color="tab:blue", lw=1.5, label=f"POS PSD (peak: {res_pos['bpm']/60.0:.2f} Hz = {res_pos['bpm']:.1f} BPM)")
    axes[2].plot(f_chrom[disp_mask], psd_chrom[disp_mask] / (np.max(psd_chrom[disp_mask]) + 1e-12),
                 color="tab:orange", lw=1.5, ls="--", label=f"CHROM PSD (peak: {res_chrom['bpm']/60.0:.2f} Hz = {res_chrom['bpm']:.1f} BPM)")

    if np.isfinite(res_pos["bpm"]):
        axes[2].axvline(res_pos["bpm"] / 60.0, color="tab:blue", ls=":", alpha=0.7)
    if np.isfinite(res_chrom["bpm"]):
        axes[2].axvline(res_chrom["bpm"] / 60.0, color="tab:orange", ls=":", alpha=0.7)

    axes[2].set_title("Normalized Frequency Spectrum (0.5 - 4.5 Hz)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Normalized PSD")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_file", nargs="?", default=None, help="path to .npz cache file")
    ap.add_argument("--latest", action="store_true", help="use the most recent cache file in data/cache/")
    ap.add_argument("--out", default=None, help="output image path (default outputs/pulse_<name>.png)")
    args = ap.parse_args()

    if args.latest:
        npz_files = sorted(DEFAULT_CACHE_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if not npz_files:
            sys.exit(f"No .npz files found in {DEFAULT_CACHE_DIR}")
        cache_path = npz_files[-1]
    elif args.cache_file:
        cache_path = Path(args.cache_file)
    else:
        ap.print_help()
        sys.exit(1)

    stem = cache_path.stem
    out_path = Path(args.out) if args.out else DEFAULT_OUTPUT_DIR / f"pulse_{stem}.png"
    plot_pulse_comparison(cache_path, out_path)


if __name__ == "__main__":
    main()
