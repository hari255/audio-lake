

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
