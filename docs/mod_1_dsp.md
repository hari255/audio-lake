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

