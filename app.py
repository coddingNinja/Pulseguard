"""Streamlit demo application (Tasks C2, C3, C4).

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="PulseGuard — Deepfake Detection", layout="wide")

st.title("PulseGuard: Deepfake Detection Through Biological Signals")
st.markdown(
    "Detecting synthetic face videos by verifying genuine physiological blood-flow (rPPG) signals."
)

tab1, tab2, tab3 = st.tabs(["Video Upload Analysis", "Live Webcam", "About / Team"])

with tab1:
    st.header("Upload Video for Analysis")
    uploaded_file = st.file_uploader("Choose an MP4 video clip", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        st.info("Analysis pipeline will run here (Task C3).")

with tab2:
    st.header("Live Webcam Pulse Extraction")
    st.info("Real-time webcam pulse extraction preview will run here (Task C2).")

with tab3:
    st.header("Team & Methodology")
    st.markdown("""
    - **Ahmad Haseeb Butt:** rPPG extraction engine (MediaPipe tracking, 16 ROIs, POS/CHROM pulse recovery)
    - **Muhammad Hamza Tahir:** Data splits, batch extraction, classifier training (XGBoost/SVM)
    - **Hafiz Hamza Ahmad:** Explainability UI, Streamlit application, evaluation reporting & slides
    """)
