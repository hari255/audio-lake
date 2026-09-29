# Module 5: LLM Compliance & Pydantic QA Agent

This module evaluates transcribed audio assets and acoustic metadata against automated compliance and quality guardrails using **Pydantic** data models.

---

## 1. Core Engineering Concepts

### Why Enforce Schema Guidance with Pydantic?
When building automated data lakehouse pipelines, free-text responses from LLMs introduce parsing risks and breaking changes in downstream services. 

* **Type Safety:** Pydantic models enforce strict data typing (booleans, bounded float ratings, string enums) for all incoming audit records.
* **Deterministic Guardrails:** Hard-coded thresholds evaluate Signal-to-Noise Ratio ($\text{SNR} \ge 15.0\text{ dB}$), Speech Density ($\ge 40.0\%$), and ASR Confidence ($\ge 60.0\%$) to assign an automated verdict before passing data to production downstream consumers.
* **Audit Trail:** Standardized JSON outputs can be directly indexed into search engines, relational databases, or Parquet lakehouse tables.

---

## 2. Compliance Audit Schema

Each record is validated against the following Pydantic model structure:

| Field Name | Type / Enum | Description |
| :--- | :--- | :--- |
| `file_name` | `string` | Name of the audited audio file |
| `contains_pii` | `boolean` | Flag indicating detection of sensitive PII |
| `pii_entities` | `List[PIIEntity]` | List of detected PII types and text strings |
| `summary` | `string` | Executive summary of transcript content |
| `sentiment` | `string` | Overall transcript sentiment (`POSITIVE`, `NEUTRAL`, `NEGATIVE`) |
| `audio_clarity_rating` | `float [0.0 - 1.0]` | Normalized acoustic clarity score derived from ASR confidence |
| `compliance_status` | `ComplianceStatus` | Pipeline verdict: `APPROVED`, `QUARANTINED`, or `REJECTED` |
| `rejection_reasons` | `List[string]` | Detailed breakdown of triggered compliance or quality rule violations |

---

## 3. Pipeline Execution Example

To evaluate an audio asset through the full pipeline and output a validated compliance audit:

```bash
python -m src.compliance_agent data/raw/2277-149896-0005.flac