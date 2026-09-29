"""
src/asr_diarization.py
----------------------
Speech-to-Text (ASR) module using Faster-Whisper.
Generates full transcripts, segment-level timestamps, and acoustic confidence scores.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from pathlib import Path
from typing import Dict, List, Union
from faster_whisper import WhisperModel


def transcribe_audio(
    file_path: Union[str, Path],
    model_size: str = "tiny",  # Options: tiny, base, small, medium, large-v3
    device: str = "cpu",       # Change to "cuda" if running on an NVIDIA GPU
    compute_type: str = "int8" # Options: int8, float16, float32
) -> Dict[str, Union[str, float, List[Dict]]]:
    """
    Transcribes an audio file using Faster-Whisper.
    
    Args:
        file_path: Path to input audio file.
        model_size: Size of Whisper model to load (default: 'tiny' for fast local CPU execution).
        device: Hardware device ('cpu' or 'cuda').
        compute_type: Quantization type ('int8', 'float16').
        
    Returns:
        Dict containing full transcript, detected language, average confidence,
        and segment-level timestamps.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # 1. Initialize Faster-Whisper model
    # The tiny model is ~75MB and loads quickly on CPU
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # 2. Run transcription pipeline
    # beam_size=5 improves accuracy by evaluating top-5 hypotheses
    segments, info = model.transcribe(
        str(file_path),
        beam_size=5,
        word_timestamps=True
    )

    # 3. Parse segments stream
    transcript_segments = []
    full_text_list = []
    total_confidence = 0.0
    segment_count = 0

    for segment in segments:
        segment_text = segment.text.strip()
        full_text_list.append(segment_text)
        
        # Segment-level confidence score (converted from log probability)
        # logprob is negative, exp(logprob) gives probability [0.0, 1.0]
        import math
        confidence = round(math.exp(segment.avg_logprob), 4)
        total_confidence += confidence
        segment_count += 1

        transcript_segments.append({
            "start_sec": round(segment.start, 2),
            "end_sec": round(segment.end, 2),
            "duration_sec": round(segment.end - segment.start, 2),
            "text": segment_text,
            "confidence": confidence
        })

    full_transcript = " ".join(full_text_list)
    avg_confidence = round(total_confidence / max(1, segment_count), 4)

    return {
        "full_transcript": full_transcript,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "average_confidence": avg_confidence,
        "segment_count": segment_count,
        "segments": transcript_segments
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/asr_diarization.py <path_to_audio_file>")
        sys.exit(1)

    sample_file = sys.argv[1]
    print(f"Transcribing audio with Faster-Whisper (tiny model): {sample_file}")

    result = transcribe_audio(sample_file, model_size="tiny", device="cpu")

    print("\n--- ASR Transcription Results ---")
    print(f"  Detected Language  : {result['language'].upper()} (Probability: {result['language_probability']})")
    print(f"  Average Confidence : {result['average_confidence'] * 100:.2f}%")
    print(f"  Full Transcript    :\n  \"{result['full_transcript']}\"\n")

    print("--- Timed Segments (First 3) ---")
    for seg in result["segments"][:3]:
        print(f"  [{seg['start_sec']:>5.2f}s -> {seg['end_sec']:>5.2f}s] (Conf: {seg['confidence'] * 100:.1f}%) : \"{seg['text']}\"")