"""Quality assessment thresholds and configuration parameters."""

# Face Detection Rate Thresholds
MIN_DETECTION_RATE_GOOD = 0.85
MIN_DETECTION_RATE_NOISY = 0.50

# SNR Thresholds (in dB)
MIN_SNR_GOOD = 2.0
MIN_SNR_NOISY = -2.0

# Quality flag literals
QUALITY_GOOD = "good"
QUALITY_NOISY = "noisy"
QUALITY_UNRELIABLE = "unreliable"
