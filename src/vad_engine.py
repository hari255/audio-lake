"""
src/vad_engine.py
-----------------
Voice Activity Detection (VAD) module using Silero VAD.
Calculates exact speech timestamps, silent intervals, and overall speech density.
"""

from pathlib import Path
from typing import Dict, List, Union
import torch
import torch.hub
import numpy as np
import librosa
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load Silero VAD from PyTorch Hub (cached locally after first download)
# We set trust_repo=True to allow torch.hub to instantiate the model
def load_vad_model():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


def process_vad(
    file_path: Union[str, Path], 
    target_sr: int = 16000, 
    threshold: float = 0.5
) -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """
    Evaluates an audio file for human voice activity.
    
    Args:
        file_path: Path to input audio file (.flac, .wav, .mp3)
        target_sr: Target sampling rate (16000 Hz expected by Silero)
        threshold: Speech probability threshold (0.0 to 1.0)
        
    Returns:
        Dict containing speech_density, total_speech_duration_sec,
        speech_segments_count, and speech_timestamps.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # 1. Load audio and resample to target_sr
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    total_duration = len(y) / sr

    if total_duration == 0:
        raise ValueError("Audio duration is 0 seconds.")

    # Convert NumPy array to PyTorch Tensor (float32)
    wav_tensor = torch.from_numpy(y).float()

    # 2. Load Silero VAD model
    model, get_speech_timestamps = load_vad_model()

    # 3. Calculate speech timestamps (returns array of dicts with start/end in seconds)
    speech_timestamps = get_speech_timestamps(
        wav_tensor,
        model,
        sampling_rate=sr,
        threshold=threshold,
        return_seconds=True
    )

    # 4. Compute Total Speech Duration
    total_speech_duration = 0.0
    formatted_timestamps = []

    for segment in speech_timestamps:
        start = round(float(segment["start"]), 3)
        end = round(float(segment["end"]), 3)
        duration = round(end - start, 3)
        
        total_speech_duration += duration
        formatted_timestamps.append({
            "start_sec": start,
            "end_sec": end,
            "duration_sec": duration
        })

    # 5. Compute Speech Density Ratio
    speech_density = float(total_speech_duration / total_duration)

    return {
        "speech_density_ratio": round(speech_density, 4),
        "total_speech_duration_sec": round(total_speech_duration, 3),
        "speech_segments_count": len(formatted_timestamps),
        "speech_timestamps": formatted_timestamps
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/vad_engine.py <path_to_audio_file>")
        sys.exit(1)

    sample_file = sys.argv[1]
    print(f"Running Silero VAD analysis on: {sample_file}")

    vad_results = process_vad(sample_file)
    print("\n--- VAD Activity Metrics ---")
    print(f"  Speech Density Ratio : {vad_results['speech_density_ratio'] * 100:.2f}%")
    print(f"  Total Speech Duration: {vad_results['total_speech_duration_sec']}s")
    print(f"  Segments Detected    : {vad_results['speech_segments_count']}")
    
    print("\n--- Detected Speech Timestamps (First 5) ---")
    for seg in vad_results["speech_timestamps"][:5]:
        print(f"  [{seg['start_sec']:>6.2f}s -> {seg['end_sec']:>6.2f}s] Duration: {seg['duration_sec']}s")