"""
src/parquet_writer.py
---------------------
Metadata persistence module using PyArrow and DuckDB.
Defines strongly-typed Parquet schemas and handles local/S3 feature store append operations.
"""

from pathlib import Path
from typing import Dict, Any, Union
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb


def get_audio_feature_schema() -> pa.Schema:
    """
    Defines the strongly-typed PyArrow schema for the audio lakehouse feature store.
    """
    return pa.schema([
        # Asset Identification
        pa.field("file_name", pa.string()),
        pa.field("file_path", pa.string()),
        
        # Module 1: DSP Signal Metrics
        pa.field("duration_sec", pa.float64()),
        pa.field("sample_rate_hz", pa.int64()),
        pa.field("peak_dbfs", pa.float64()),
        pa.field("rms_dbfs", pa.float64()),
        pa.field("snr_db", pa.float64()),
        pa.field("clipping_ratio", pa.float64()),
        
        # Module 2: Voice Activity Detection (VAD)
        pa.field("speech_density_ratio", pa.float64()),
        pa.field("total_speech_duration_sec", pa.float64()),
        pa.field("speech_segments_count", pa.int64()),
        
        # Module 3: ASR Transcription & Acoustics
        pa.field("language", pa.string()),
        pa.field("language_probability", pa.float64()),
        pa.field("average_confidence", pa.float64()),
        pa.field("full_transcript", pa.string()),
    ])


def append_record_to_parquet(
    record_data: Dict[str, Any], 
    output_parquet_path: Union[str, Path]
) -> None:
    """
    Appends a single audio pipeline record to a columnar Parquet feature store file.
    
    Args:
        record_data: Combined dictionary containing output from DSP, VAD, and ASR modules.
        output_parquet_path: Destination path for the .parquet feature table.
    """
    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    schema = get_audio_feature_schema()

    # Format record dictionary to conform with schema types
    formatted_data = {
        key: [record_data.get(key)] for key in schema.names
    }

    # Convert to PyArrow Table
    new_table = pa.Table.from_pydict(formatted_data, schema=schema)

    # If file exists, append table; otherwise write new Parquet file
    if output_path.exists():
        existing_table = pq.read_table(output_path)
        combined_table = pa.concat_tables([existing_table, new_table])
        pq.write_table(combined_table, output_path, compression="snappy")
    else:
        pq.write_table(new_table, output_path, compression="snappy")


def query_feature_store_duckdb(parquet_path: Union[str, Path]) -> None:
    """
    Executes high-speed DuckDB SQL queries directly over the Parquet feature store.
    """
    parquet_path = str(parquet_path)
    con = duckdb.connect()

    print("\n--- DuckDB SQL Query: High Quality Speech Signals (SNR > 20 dB & Speech Density > 80%) ---")
    query = f"""
        SELECT 
            file_name,
            duration_sec,
            snr_db,
            speech_density_ratio,
            average_confidence,
            full_transcript
        FROM read_parquet('{parquet_path}')
        WHERE snr_db > 20.0 AND speech_density_ratio > 0.80
    """
    df = con.execute(query).fetchdf()
    print(df.to_string(index=False))

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root directory to sys.path
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.dsp_metrics import analyze_audio_dsp
    from src.vad_engine import process_vad
    from src.asr_diarization import transcribe_audio

    if len(sys.argv) < 2:
        print("Usage: python src/parquet_writer.py <path_to_audio_file>")
        sys.exit(1)

    sample_file = sys.argv[1]
    feature_store_path = Path("data/feature_store/audio_metadata.parquet")

    print(f"Running full extraction pipeline on: {sample_file}")

    # Extract signals from Modules 1, 2, and 3
    dsp = analyze_audio_dsp(sample_file)
    vad = process_vad(sample_file)
    asr = transcribe_audio(sample_file, model_size="tiny")

    # Merge payloads
    record = {
        "file_name": Path(sample_file).name,
        "file_path": str(sample_file),
        **dsp,
        **vad,
        **asr
    }

    # Persist to Parquet
    append_record_to_parquet(record, feature_store_path)
    print(f"Successfully appended record to Feature Store: {feature_store_path}")

    # Query with DuckDB
    query_feature_store_duckdb(feature_store_path)