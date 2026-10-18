"""A5: Validation script for the UBFC-rPPG benchmark dataset.

Usage:
    python scripts/validate_ubfc.py --data-dir data/ubfc --method pos
    python scripts/validate_ubfc.py --synthetic-selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import csv
from scipy.stats import pearsonr

from pulseguard.pulse import recover_pulse
from pulseguard.roi import extract_rois_ex


def parse_ground_truth(gt_path: Path) -> tuple[float, np.ndarray]:
    """Parse UBFC ground_truth.txt.

    UBFC dataset provides ground_truth.txt with 3 rows:
        Row 0: Trace / PPG signal values
        Row 1: Heart rate (HR) in BPM per time step
        Row 2: Timestamp in seconds
    Some versions format as comma or space separated numbers.

    Returns:
        (mean_hr_bpm, hr_series)
    """
    with open(gt_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) >= 2:
        # Row 1 is the heart rate series
        hr_tokens = lines[1].replace(",", " ").split()
        hr_vals = np.array([float(x) for x in hr_tokens if x.replace(".", "", 1).isdigit()], dtype=np.float64)
    else:
        # Single column format fallback
        all_tokens = " ".join(lines).replace(",", " ").split()
        hr_vals = np.array([float(x) for x in all_tokens], dtype=np.float64)

    valid_hr = hr_vals[(hr_vals >= 40.0) & (hr_vals <= 220.0)]
    if len(valid_hr) == 0:
        mean_hr = float(np.mean(hr_vals)) if len(hr_vals) > 0 else 72.0
    else:
        mean_hr = float(np.mean(valid_hr))

    return mean_hr, hr_vals


def run_synthetic_selftest(method: str = "pos"):
    """Verify evaluation logic using synthetic data."""
    print("=== UBFC Validation: Synthetic Self-Test Mode ===")
    print("NOTE: This uses synthetically generated pulse data to verify evaluation metrics.")
    print("      It is NOT a benchmark validation on real subjects.\n")

    from tests.test_pulse import generate_synthetic_rois

    results = []
    ground_truth_bpm_list = [54.0, 60.0, 72.0, 75.0, 80.0, 90.0, 100.0, 110.0, 120.0, 125.0]

    for idx, gt_bpm in enumerate(ground_truth_bpm_list):
        subj_name = f"synth_subj_{idx+1:02d}"
        target_hz = gt_bpm / 60.0
        rois = generate_synthetic_rois(fps=30.0, duration=15.0, target_hz=target_hz, seed=idx + 10)
        rec = recover_pulse(rois, 30.0, method=method)
        est_bpm = rec["bpm"]
        err = abs(est_bpm - gt_bpm)
        results.append({
            "subject": subj_name,
            "gt_bpm": gt_bpm,
            "est_bpm": round(est_bpm, 2),
            "abs_error": round(err, 2),
        })

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ubfc_synthetic_selftest.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "gt_bpm", "est_bpm", "abs_error"])
        writer.writeheader()
        writer.writerows(results)

    gt_list = [r["gt_bpm"] for r in results]
    est_list = [r["est_bpm"] for r in results]
    err_list = [r["abs_error"] for r in results]

    mae = float(np.mean(err_list))
    rmse = float(np.sqrt(np.mean(np.array(err_list) ** 2)))
    r_val, _ = pearsonr(gt_list, est_list)

    print(f"{'Subject':<18} {'GT BPM':<10} {'Est BPM':<10} {'Abs Err':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['subject']:<18} {r['gt_bpm']:<10.2f} {r['est_bpm']:<10.2f} {r['abs_error']:<10.2f}")

    print("\n--- Summary Metrics ---")
    print(f"Subjects evaluated: {len(results)}")
    print(f"MAE:                {mae:.2f} BPM  (Target: <= 5.0 BPM -> {'PASS' if mae <= 5.0 else 'FAIL'})")
    print(f"RMSE:               {rmse:.2f} BPM")
    print(f"Pearson r:          {r_val:.4f}")
    print(f"Results saved to:   {out_csv}")


def validate_ubfc(data_dir: Path, method: str = "pos", max_subjects: int = None):
    """Run validation across real UBFC-rPPG subjects."""
    if not data_dir.exists():
        sys.exit(
            f"\n[ERROR] UBFC dataset directory '{data_dir}' not found.\n"
            f"To run real benchmark validation (Task A5):\n"
            f"1. Download UBFC-rPPG from: https://sites.google.com/view/ybvision/ubfc-rppg\n"
            f"2. Extract subject folders (subject1/, subject2/...) into: data/ubfc/\n"
            f"3. Run: python scripts/validate_ubfc.py --data-dir data/ubfc --method {method}\n"
            f"\nTo verify evaluation script logic without real data, run:\n"
            f"   python scripts/validate_ubfc.py --synthetic-selftest\n"
        )

    # Find subject directories containing both video and ground_truth.txt
    subdirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir() and (p / "ground_truth.txt").exists()]
    if not subdirs:
        sys.exit(
            f"\n[ERROR] No valid UBFC subjects found in '{data_dir}'.\n"
            f"Expected subdirectories containing 'ground_truth.txt' and a video file ('vid.avi').\n"
        )

    if max_subjects is not None:
        subdirs = subdirs[:max_subjects]

    print(f"=== Running UBFC Validation ({len(subdirs)} subjects, method: {method.upper()}) ===")
    results = []

    for subj_path in subdirs:
        gt_path = subj_path / "ground_truth.txt"
        vid_candidates = list(subj_path.glob("*.avi")) + list(subj_path.glob("*.mp4"))
        if not vid_candidates:
            print(f"Skipping {subj_path.name}: no video file found.")
            continue
        vid_path = vid_candidates[0]

        gt_bpm, _ = parse_ground_truth(gt_path)

        try:
            res = extract_rois_ex(vid_path, use_cache=True)
            pulse_res = recover_pulse(res.rois, res.fps, method=method)
            est_bpm = pulse_res["bpm"]
            err = abs(est_bpm - gt_bpm)

            results.append({
                "subject": subj_path.name,
                "gt_bpm": round(gt_bpm, 2),
                "est_bpm": round(est_bpm, 2),
                "abs_error": round(err, 2),
            })
            print(f"  {subj_path.name:<15}: GT={gt_bpm:5.1f} | Est={est_bpm:5.1f} | Err={err:4.1f} BPM")
        except Exception as exc:
            print(f"  {subj_path.name:<15}: Failed ({exc})")

    if not results:
        sys.exit("No subjects were successfully evaluated.")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ubfc_errors.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "gt_bpm", "est_bpm", "abs_error"])
        writer.writeheader()
        writer.writerows(results)

    gt_list = [r["gt_bpm"] for r in results]
    est_list = [r["est_bpm"] for r in results]
    err_list = [r["abs_error"] for r in results]

    mae = float(np.mean(err_list))
    rmse = float(np.sqrt(np.mean(np.array(err_list) ** 2)))
    r_val, _ = pearsonr(gt_list, est_list) if len(results) > 1 else (np.nan, np.nan)

    print("\n--- Final UBFC Benchmark Results ---")
    print(f"Subjects evaluated: {len(results)}")
    print(f"MAE:                {mae:.2f} BPM  (Target: <= 5.0 BPM -> {'PASS' if mae <= 5.0 else 'FAIL'})")
    print(f"RMSE:               {rmse:.2f} BPM")
    print(f"Pearson r:          {r_val:.4f}")
    print(f"Results saved to:   {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/ubfc", help="path to UBFC dataset root")
    ap.add_argument("--method", choices=["pos", "chrom"], default="pos", help="recovery algorithm")
    ap.add_argument("--max-subjects", type=int, default=None, help="limit number of subjects")
    ap.add_argument("--synthetic-selftest", action="store_true", help="run evaluation self-test on synthetic data")
    args = ap.parse_args()

    if args.synthetic_selftest:
        run_synthetic_selftest(method=args.method)
    else:
        validate_ubfc(Path(args.data_dir), method=args.method, max_subjects=args.max_subjects)


if __name__ == "__main__":
    main()
