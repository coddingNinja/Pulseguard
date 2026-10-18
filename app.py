"""Owner: Hafiz Hamza Ahmad
Task ID: C2, C3, C4 / Streamlit Demo Application
Acceptance Criteria:
    - Tab 1: Video upload mode returning real/fake score, quality flag, pulse waveform, and heatmap.
    - Tab 2: Live webcam mode with real-time feedback and quality indicator.
    - Tab 3: Methodology and architecture explanation.
    - Works with mock data out of the box until model weights land.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pulseguard.coherence import coherence_features, coherence_matrix
from pulseguard.predict import predict
from pulseguard.roi import REGION_NAMES
from pulseguard.viz import plot_heatmap, plot_waveform
from scripts.make_mock_data import make_mock_signal_dict

st.set_page_config(page_title="PulseGuard — Deepfake Detection", layout="wide")

st.title("🛡️ PulseGuard: Deepfake Detection Through Biological Signals")
st.markdown(
    "Detecting synthetic face videos by verifying genuine physiological blood-flow (**rPPG**) signals."
)

tab_upload, tab_live, tab_about = st.tabs(["📁 Video Upload Analysis", "📷 Live Webcam", "ℹ️ About & Team"])

with tab_upload:
    st.header("Video Upload Analysis")
    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("Choose a video file (.mp4, .avi)", type=["mp4", "avi", "mov"])
        use_mock = st.checkbox("Use Mock Pipeline Demo Data", value=True)
        analyze_btn = st.button("Run Deepfake Analysis", type="primary")

    with col2:
        if analyze_btn:
            if use_mock or uploaded_file is None:
                st.info("Displaying Mock Analysis Data (Task C1/C3 Scaffolding)")
                mock_type = st.radio("Simulate Sample Type:", ["Genuine Face Video", "Deepfake Video"], horizontal=True)
                is_real = mock_type == "Genuine Face Video"

                sig_dict = make_mock_signal_dict(is_real=is_real, duration=15.0)
                coh_mat = coherence_matrix(sig_dict["signals"])
                score = 0.94 if is_real else 0.12
                quality = "good"
            else:
                temp_path = Path("data/raw") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Extracting rPPG signals and computing features..."):
                    pred_res = predict(temp_path)
                    score = pred_res["score"]
                    quality = pred_res["quality_flag"]
                    sig_dict = {"signals": pred_res["signals"], "bpm": pred_res["bpm"], "fps": 30.0, "overall": np.mean(pred_res["signals"], axis=0)}
                    coh_mat = coherence_matrix(sig_dict["signals"])

            # Metric Cards
            m1, m2, m3 = st.columns(3)
            verdict = "REAL FACE" if score >= 0.5 else "SYNTHETIC / DEEPFAKE"
            verdict_color = "green" if score >= 0.5 else "red"

            m1.metric("Authenticity Score", f"{score * 100:.1f}%", f":{verdict_color}[{verdict}]")
            m2.metric("Detected Pulse (BPM)", f"{sig_dict.get('bpm', 0.0):.1f}")
            m3.metric("Signal Quality", quality.upper())

            # Visualizations
            st.subheader("Biological Evidence Views")
            v1, v2 = st.columns([3, 2])
            with v1:
                fig_wave = plot_waveform(sig_dict, fps=sig_dict.get("fps", 30.0))
                st.pyplot(fig_wave)
            with v2:
                fig_heat = plot_heatmap(coh_mat, REGION_NAMES)
                st.pyplot(fig_heat)

with tab_live:
    st.header("Live Webcam Mode")
    st.markdown("""
    To launch the high-performance local OpenCV webcam monitor with live POS/CHROM pulse graphing:
    ```bash
    python scripts/webcam_roi.py --method pos
    ```
    - **Warmup:** ~10-15 seconds for exposure and pulse filter stabilization.
    - **Controls:** Press `q` to quit and cache traces, press `s` to save an annotated snapshot.
    """)

with tab_about:
    st.header("About PulseGuard")
    st.markdown("""
    ### Team
    - **Ahmad Haseeb Butt (A):** rPPG Signal Engine (MediaPipe Face Mesh, 16 ROIs, POS/CHROM pulse recovery, spectral features)
    - **Muhammad Hamza Tahir (B):** Data pipeline, splits, batch extraction, classifier training (XGBoost/SVM)
    - **Hafiz Hamza Ahmad (C):** Explainability visualizations, Streamlit application, evaluation report & slides

    ### Methodology
    1. **Face Tracking (A1):** MediaPipe Face Mesh extracts 468 facial points per frame.
    2. **ROI Colour Extraction (A2):** 16 symmetric facial regions extracted and averaged.
    3. **rPPG Recovery (A3):** Plane-Orthogonal-to-Skin (POS) & Chrominance (CHROM) algorithms with 0.7-4.0 Hz bandpass filtering.
    4. **Feature & Coherence Analysis (A4/A6):** Power spectral density, SNR, and cross-region coherence matrix.
    5. **Classification (B4):** Trained classifier on handcrafted biological features.
    """)
