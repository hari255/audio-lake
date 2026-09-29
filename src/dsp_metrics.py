"""
src/dsp_metrics.py
------------------
Digital Signal Processing (DSP) module for evaluating raw audio quality.
Computes RMS Energy, Peak dBFS, Signal-to-Noise Ratio (SNR), and Digital Clipping Ratio.
"""

from pathlib import Path
from typing import Dict, Union
import librosa
import numpy as np


def analyze_audio_dsp(
    file_path: Union[str, Path], 
    target_sr: int = 16000, 
    clipping_threshold: float = 0.99
) -> Dict[str, float]:
    """
    Loads an audio file and calculates signal quality metrics.
    
    Args:
        file_path: Path to the input audio file (.flac, .wav, .mp3).
        target_sr: Target sample rate for resampling (default: 16000 Hz).
        clipping_threshold: Absolute amplitude value considered clipping (default: 0.99).
        
    Returns:
        Dict containing duration, sample rate, peak_dbfs, rms_dbfs, snr_db, and clipping_ratio.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found at: {file_path}")

    # 1. Load and resample audio into a 1D NumPy array [-1.0, 1.0]
    # mono=True mixes multi-channel audio to a single mono stream
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    if len(y) == 0:
        raise ValueError(f"Loaded audio array is empty: {file_path}")

    # 2. Basic Duration & Sample Metrics
    duration_sec = float(len(y) / sr)

    # 3. Peak Amplitude & Peak dBFS
    peak_amplitude = float(np.max(np.abs(y)))
    # Avoid log10(0) by adding a tiny epsilon value (1e-10)
    peak_dbfs = float(20 * np.log10(peak_amplitude + 1e-10))

    # 4. RMS Energy & RMS dBFS
    rms = float(np.sqrt(np.mean(y ** 2)))
    rms_dbfs = float(20 * np.log10(rms + 1e-10))

    # 5. Digital Clipping Ratio
    # Calculates percentage of samples that hit or exceed clipping_threshold
    clipped_samples = np.sum(np.abs(y) >= clipping_threshold)
    clipping_ratio = float(clipped_samples / len(y))

    # 6. Signal-to-Noise Ratio (SNR) Estimation
    # Compute frame-level RMS energy (frame size = 20ms at 16kHz -> 320 samples)
    frame_length = int(sr * 0.02)
    hop_length = int(frame_length / 2)
    frames_rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Estimate noise from the quietest 10% of frames; speech from the top 10%
    sorted_frames = np.sort(frames_rms)
    noise_floor_rms = np.mean(sorted_frames[: max(1, int(len(sorted_frames) * 0.10))])
    speech_peak_rms = np.mean(sorted_frames[-max(1, int(len(sorted_frames) * 0.10)):])

    # SNR in decibels
    snr_db = float(10 * np.log10((speech_peak_rms ** 2 + 1e-10) / (noise_floor_rms ** 2 + 1e-10)))

    return {
        "duration_sec": round(duration_sec, 3),
        "sample_rate_hz": sr,
        "total_samples": len(y),
        "peak_dbfs": round(peak_dbfs, 2),
        "rms_dbfs": round(rms_dbfs, 2),
        "snr_db": round(snr_db, 2),
        "clipping_ratio": round(clipping_ratio, 6),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/dsp_metrics.py <path_to_audio_file>")
        sys.exit(1)

    sample_file = sys.argv[1]
    print(f"Analyzing signal quality for: {sample_file}")
    
    metrics = analyze_audio_dsp(sample_file)
    print("\n--- DSP Signal Quality Metrics ---")
    for key, value in metrics.items():
        print(f"  {key:<18}: {value}")