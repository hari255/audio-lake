
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