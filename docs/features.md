# A4 & A6: Feature Engineering & Physiological Coherence

PulseGuard extracts biological signals from facial videos to differentiate real humans from synthetic deepfakes. This document outlines the 72 handcrafted features and physiological principles.

---

## 1. Physiological Principle

Human cardiac cycles pump blood periodically through the capillary beds under facial dermis. This produces microscopic optical absorption fluctuations known as **remote photoplethysmography (rPPG)**:
- **Real Faces:** Display clean cardiac pulsations across all facial regions, with strong cross-regional correlation and uniform pulse rate.
- **Deepfakes / Synthetics:** Generative models synthesize visual appearance frame-by-frame or blur boundaries. They fail to preserve micro-color periodicity, exhibiting low signal-to-noise ratios, inconsistent regional frequencies, and disordered cross-regional correlations.

---

## 2. Feature Vector Specification (`FEATURE_NAMES`)

`compute_features(signal_dict)` produces a fixed-order 1D vector of **72 numerical features**:

### A. Global Spectral Features (4 features)
1. `overall_bpm`: Peak frequency across 0.7–4.0 Hz (42–240 BPM) multiplied by 60.
2. `overall_snr`: Power inside $\pm 0.1$ Hz of the peak and 2nd harmonic divided by the remaining power inside 0.7–4.0 Hz (in dB).
3. `overall_peak_power`: Welch Power Spectral Density at the cardiac fundamental peak.
4. `overall_spectral_peak_ratio`: Ratio of maximum spectral peak power to total in-band energy.

### B. Cross-Region Spatial Coherence Features (4 features - Task A6)
1. `coherence_mean`: Mean off-diagonal pairwise Pearson correlation across all 16 facial region signals.
2. `coherence_std`: Standard deviation of the off-diagonal correlation coefficients (measures spatial uniformity).
3. `coherence_min`: Minimum off-diagonal correlation among valid skin regions.
4. `bpm_agreement_ratio`: Proportion of the 16 regions whose locally estimated BPM falls within $\pm 5.0$ BPM of the global estimated BPM.

### C. Per-Region Spectral & Temporal Features (16 regions $\times$ 4 features = 64 features)
For each of the 16 regions in `pulseguard.roi.REGION_NAMES` (`forehead_c`, `cheek_r_upper`, etc.):
1. `<region>_bpm`: Local heart rate estimate.
2. `<region>_snr`: Local signal-to-noise ratio in dB.
3. `<region>_peak_power`: Local spectral peak power.
4. `<region>_variance`: Variance of the local band-passed pulse waveform.

---

## 3. Coherence Matrix Formulation

The $16 \times 16$ correlation matrix $M$ is defined as:
$$M_{i, j} = \frac{\sum_{t=1}^T (s_i(t) - \bar{s}_i)(s_j(t) - \bar{s}_j)}{\sqrt{\sum_{t=1}^T (s_i(t) - \bar{s}_i)^2 \sum_{t=1}^T (s_j(t) - \bar{s}_j)^2}}$$

For genuine faces, $M_{i, j} \approx 0.7 - 0.95$ across forehead, nose, and cheeks. For deepfakes, $M_{i, j} \approx 0.0 - 0.3$.
