

# Module 4: Apache Parquet Metadata & Feature Store Writer

This module persists extracted signal metadata (DSP metrics, Voice Activity Detection indicators, and Speech-to-Text transcripts) into a strongly-typed, columnar **Apache Parquet** feature store using **PyArrow** and **DuckDB**.

---

## 4.1. Core Engineering Concepts

### Why Parquet for Audio Lakehouses?
In enterprise unstructured data architectures, storing metadata as loose JSON or in relational databases creates severe performance bottlenecks at scale. Columnar storage formats like **Apache Parquet** provide:

* **Column Pruning:** Analytical engines read only requested columns (e.g., scanning `snr_db` across 1,000,000 files without deserializing transcript strings).
* **Strong Type Enforcement:** `pyarrow.Schema` guarantees explicit data types (float64, int64, strings) across all pipeline records.
* **Storage Efficiency:** Built-in Snappy or ZSTD compression drastically reduces storage footprint on object stores like AWS S3 or local disk.
* **SQL Queryability:** Enables zero-copy, high-speed analytical queries via **DuckDB** or **Polars**.

---

## 4.2. Feature Store Schema Design

The feature store conforms to a unified schema combining signals from Modules 1, 2, and 3:

| Field Name | PyArrow Type | Description / Source |
| :--- | :--- | :--- |
| `file_name` | `string` | Audio filename (e.g., `2277-149896-0005.flac`) |
| `file_path` | `string` | Absolute disk or S3 path |
| `duration_sec` | `float64` | Total audio length in seconds |
| `sample_rate_hz` | `int64` | Audio sample rate (e.g., 16000 Hz) |
| `peak_dbfs` | `float64` | Peak decibel volume level relative to full scale |
| `rms_dbfs` | `float64` | Average RMS energy in dBFS |
| `snr_db` | `float64` | Signal-to-Noise Ratio in dB (Module 1) |
| `clipping_ratio` | `float64` | Proportion of clipped samples $\vert{}y\vert{} \ge 0.99$ |
| `speech_density_ratio` | `float64` | Proportion of total audio containing voice (Module 2) |
| `total_speech_duration_sec` | `float64` | Cumulative duration of speech segments |
| `speech_segments_count` | `int64` | Number of distinct VAD speech intervals |
| `language` | `string` | Primary detected ISO language code (Module 3) |
| `language_probability` | `float64` | Confidence score for detected language |
| `average_confidence` | `float64` | ASR model acoustic confidence score |
| `full_transcript` | `string` | Complete transcribed text output |

---

## 4.3. Pipeline Execution & Querying

To process an audio asset through all three modules and append the structured record into the Parquet Feature Store:

```bash
python -m src.parquet_writer data/raw/2277-149896-0005.flac