## Master End-to-End Batch Pipeline Orchestrator for audio-lake.
## Processes all audio assets through Modules 1-5 and persists features to Parquet.


import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dsp_metrics import analyze_audio_dsp
from src.vad_engine import process_vad
from src.asr_diarization import transcribe_audio
from src.parquet_writer import append_record_to_parquet, query_feature_store_duckdb
from src.compliance_agent import evaluate_compliance_rule_based


def process_audio_lake_batch(raw_dir: str, feature_store_path: str) -> None:
    raw_path = Path(raw_dir)
    audio_files = list(raw_path.glob("*.flac")) + list(raw_path.glob("*.wav"))

    if not audio_files:
        print(f"No audio files found in {raw_dir}")
        return

    print(f"--- Launching audio-lake Pipeline: Processing {len(audio_files)} files ---\n")

    for file in audio_files:
        print(f" Processing: {file.name}")
        
        # Module 1: DSP Metrics
        dsp = analyze_audio_dsp(file)
        
        # Module 2: VAD Speech Density
        vad = process_vad(file)
        
        # Module 3: ASR Transcription
        asr = transcribe_audio(file, model_size="tiny")
        
        # Module 5: Compliance Audit Report
        combined = {
            "file_name": file.name,
            "file_path": str(file),
            **dsp,
            **vad,
            **asr
        }
        report = evaluate_compliance_rule_based(combined)
        
        # Module 4: Parquet Persistence
        append_record_to_parquet(combined, feature_store_path)
        
        print(f"   Status: [{report.compliance_status.value}] | SNR: {dsp['snr_db']:.1f} dB | Speech Density: {vad['speech_density_ratio']*100:.1f}%\n")

    print(f"Batch processing complete. Feature store updated at: {feature_store_path}")
    
    # Query summary using DuckDB
    query_feature_store_duckdb(feature_store_path)


if __name__ == "__main__":
    raw_directory = "data/raw"
    feature_store = "data/feature_store/audio_metadata.parquet"
    
    process_audio_lake_batch(raw_directory, feature_store)