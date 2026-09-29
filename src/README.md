# Module 1: Core Audio Engineering Concepts

This document outlines the fundamental Digital Signal Processing (DSP) and acoustics concepts used to evaluate audio data quality prior to downstream machine learning or ASR model processing.

---

## 1.1. Digital Audio as Numerical Signal Arrays

When a microphone records sound, it measures continuous variations in air pressure and converts them into digital representations:

* **Sample Rate ($f_s$):** The frequency at which continuous audio amplitude is sampled per second. For speech processing pipelines, **$16,000\text{ Hz}$** (16 kHz) is standard, meaning $16,000$ floating-point data points represent 1 second of audio.
* **Amplitude ($y$):** A 1D array of normalized floating-point numbers typically bounded between **$-1.0$** and **$+1.0$**.

---

## 1.2. Root Mean Square (RMS) Energy & Decibels (dBFS)

The overall loudness and signal power of an audio waveform are calculated using the **Root Mean Square (RMS)** energy:

$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} y_i^2}$$

Because human perception of loudness is logarithmic rather than linear, amplitude values are mapped to **Decibels relative to Full Scale ($\text{dBFS}$)**:

$$\text{dBFS} = 20 \cdot \log_{10}(\text{RMS})$$

* **Peak dBFS:** Measures the maximum volume point in the signal. $0.0\text{ dBFS}$ represents maximum digital capacity.
* **RMS dBFS:** Measures average signal intensity across the entire audio duration.

---

## 1.3. Signal-to-Noise Ratio (SNR)

**Signal-to-Noise Ratio (SNR)** quantifies how clearly the target speech signal stands out against background noise floor interference:

$$\text{SNR}_{\text{dB}} = 10 \cdot \log_{10}\left(\frac{P_{\text{speech}}}{P_{\text{noise}}}\right)$$

### Pipeline Estimation Method:
1. Divide audio into 20ms overlapping frames.
2. Calculate RMS energy for each individual frame.
3. Estimate $P_{\text{noise}}$ from the quietest $10\%$ of frames (noise floor).
4. Estimate $P_{\text{speech}}$ from the loudest $10\%$ of frames (speech peaks).

### Interpretation Thresholds:
* **$\text{SNR} > 20\text{ dB}$:** High-quality audio suitable for speech recognition (ASR) and feature stores.
* **$\text{SNR} < 10\text{ dB}$:** High noise floor; likely requires noise suppression or automated quarantine.

---

## 1.4. Digital Clipping Ratio

**Digital Clipping** occurs when the analog audio input exceeds the maximum capacity of the Analog-to-Digital Converter (ADC), resulting in clipped amplitude peaks at $|y| \ge 0.99$.

* Clipping distorts the audio waveform, introduces harsh harmonic distortion, and degrades speech model inference accuracy.
* **Clipping Ratio** measures the proportion of total samples exceeding the threshold:

$$\text{Clipping Ratio} = \frac{\text{Count}(|y| \ge 0.99)}{\text{Total Samples}}$$

Audio files with a clipping ratio $> 0.01$ ($1\%$) indicate gain distortion during recording.



# Module 2: Voice Activity Detection (VAD) & Speech Density

This module isolates active human speech from silence or background noise using **Silero VAD**, computing speech timestamps and calculating the overall **Speech Density Ratio**.

---

## 2.1. Core Engineering Concepts

### Why Pre-Filter with VAD?
Raw enterprise audio often contains significant silence, breathing, or background hums. Passing unsegmented silence to heavy speech-to-text (ASR) Transformers or LLMs wastes memory and processing time.

* **Speed & Cost:** Silero VAD runs on CPU at **under 1 ms per audio frame** ($30\text{ms}$ / 512 samples at 16 kHz).
* **Automated Quarantine:** Files with little to no voice activity can be flagged and dropped early in the ETL pipeline.

---

## 2.2. Key Metrics & Formulas

### Speech Density Ratio
Calculates the proportion of the total audio duration containing human voice:

$$\text{Speech Density} = \frac{\sum \text{Duration}(\text{Active Speech Segments})}{\text{Total Audio Duration}}$$

* **Speech Density $< 0.10$ ($10\%$):** Low speech activity; candidate for quarantine.
* **Speech Density $> 0.70$ ($70\%$):** High density continuous audio stream.

### Frame-Level Speech Timestamps
Returns millisecond-level arrays marking speech boundaries:

$$\text{Segment}_i = \{ \text{start\_sec}: t_{\text{start}}, \, \text{end\_sec}: t_{\text{end}}, \, \text{duration\_sec}: t_{\text{end}} - t_{\text{start}} \}$$

---

## 2.3. Pipeline Implementation Summary

1. **Load & Resample:** Load input audio and resample to $16,000\text{ Hz}$ mono float32 tensor.
2. **Probability Inference:** Evaluate $30\text{ms}$ audio chunks with `silero-vad` model weights ($threshold = 0.5$).
3. **Chunk Mapping:** Group contiguous speech frames into discrete intervals.
4. **Metadata Metrics:** Aggregate active speech duration and compute `speech_density_ratio`.

---

## 2.4. Pipeline Execution Example

```bash
python src/vad_engine.py data/raw/2277-149896-0005.flac

```

# Module 3: Acoustic Modeling & Speech-to-Text (ASR)

This module converts processed audio into text transcripts using **Faster-Whisper**, extracting segment-level millisecond timestamps, language detection probabilities, and acoustic confidence scores.

---

## 3.1. Core Engineering Concepts

### Why Faster-Whisper (CTranslate2)?
Standard transformer-based ASR models like OpenAI Whisper can be computationally expensive and slow when run in production pipelines. `faster-whisper` re-implements Whisper using **CTranslate2**, a custom C++ inference engine for Transformer models.

* **Inference Speed:** Up to **4x faster** than vanilla PyTorch implementations.
* **8-bit Quantization (`int8`):** Reduces memory usage dramatically while preserving transcription accuracy.
* **Acoustic Confidence:** Computes segment-level confidence probabilities derived from model log probabilities ($\text{Probability} = e^{\text{avg\_logprob}}$).

---

## 3.2. Key Metrics & Pipeline Outputs

### Acoustic Confidence Score
Transforms logarithmic probability output into a normalized score ($0.0 \text{ to } 1.0$):

$$\text{Confidence} = e^{\text{avg\_logprob}}$$

* **Confidence $> 0.85$:** High accuracy; ready for downstream LLM evaluation.
* **Confidence $< 0.50$:** Low acoustic clarity; flag for manual review or secondary ASR pass.

### Segment Timestamps Payload
Extracts frame-accurate time boundaries for every recognized phrase:

$$\text{Segment}_i = \{ \text{start\_sec}: t_{\text{start}}, \, \text{end\_sec}: t_{\text{end}}, \, \text{text}: \text{"..."}, \, \text{confidence}: c \}$$

---

## 3.3. Pipeline Implementation Summary

1. **Model Initialization:** Instantiates CTranslate2-backed Whisper engine (`tiny`, `base`, or `small` model sizes).
2. **Decoding & Beam Search:** Runs transcription using beam search decoding ($\text{beam\_size} = 5$) to evaluate optimal text candidate hypotheses.
3. **Timestamp Alignment:** Extracts word and segment boundaries mapped to absolute audio time.
4. **Metadata Aggregation:** Calculates average acoustic confidence across all segments and detects the primary language.

---

## 3.4. Pipeline Execution Example

```bash
python src/asr_diarization.py data/raw/2277-149896-0005.flac
```

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