# AUDIO-LAKE


**`audio-lake`** is an end-to-end unstructured audio ingestion and AI feature engineering pipeline built for scalable speech data lakehouses. It processes raw audio streams through signal processing quality checks, voice activity detection, automatic speech recognition, and structured metadata persistence.

### Key Capabilities
* **Signal Quality Engineering:** Calculates DSP metrics (SNR, RMS energy dBFS, peak dBFS, digital clipping) to assess raw audio health.
* **Voice Activity Detection (VAD):** Uses Silero VAD to isolate active human speech, compute speech density ratios, and extract millisecond-level timestamps.
* **Accelerated Speech-to-Text (ASR):** Transcribes audio using `faster-whisper` (CTranslate2 `int8` quantization) to produce transcripts, segment timestamps, and acoustic confidence scores.
* **Columnar Feature Store:** Persists strongly-typed PyArrow schemas to Snappy-compressed Apache Parquet tables, enabling zero-copy SQL analytical queries via DuckDB.
* **Compliance & QA Guardrails:** Evaluates transcripts and acoustic metrics against Pydantic models to detect PII and assign automated pipeline verdicts (`APPROVED`, `QUARANTINED`, `REJECTED`).

## ARCHITECTURE OVERVIEW

                        +-------------+-------------+
                        |  Raw Unstructured Audio   |
                        |   (.flac, .wav, .mp3)     |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |   Module 1: DSP Engine    |
                        |   (SNR, RMS, Clipping)    |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        |   Module 2: Silero VAD    |
                        |  (Speech Density Ratio)   |
                        +-------------+-------------+
                                      |
                                      v
                        +---------------------------+
                        | Module 3: Faster-Whisper  |
                        |  (ASR & Timestamps)       |
                        +-------------+-------------+
                                      |
                  +-------------------+-------------------+
                  |                                       |
                  v                                       v
    +---------------------------+           +---------------------------+
    | Module 4: Parquet Store   |           | Module 5: QA & Compliance |
    |   (PyArrow + DuckDB)      |           |    (Pydantic Models)      |
    +---------------------------+           +---------------------------+
