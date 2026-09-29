"""
src/compliance_agent.py
-----------------------
LLM Compliance & QA Agent module using Pydantic.
Evaluates transcripts for compliance, PII presence, sentiment, and pipeline quality guardrails.
"""

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


class ComplianceStatus(str, Enum):
    APPROVED = "APPROVED"
    QUARANTINED = "QUARANTINED"
    REJECTED = "REJECTED"


class PIIEntity(BaseModel):
    entity_type: str = Field(description="Type of PII detected (e.g., NAME, PHONE, ADDRESS, SSN, CREDIT_CARD)")
    detected_text: str = Field(description="The actual text string flagged as potential PII")


class AudioComplianceReport(BaseModel):
    file_name: str = Field(description="Name of the audio file being audited")
    contains_pii: bool = Field(description="True if sensitive personally identifiable information is detected")
    pii_entities: List[PIIEntity] = Field(default_factory=list, description="List of detected PII entities")
    summary: str = Field(description="Brief 1-2 sentence executive summary of the spoken content")
    sentiment: str = Field(description="Overall sentiment of the audio: POSITIVE, NEUTRAL, or NEGATIVE")
    audio_clarity_rating: float = Field(ge=0.0, le=1.0, description="Overall transcript clarity rating between 0.0 and 1.0")
    compliance_status: ComplianceStatus = Field(description="Final pipeline verdict: APPROVED, QUARANTINED, or REJECTED")
    rejection_reasons: List[str] = Field(default_factory=list, description="List of reasons if flagged or rejected")


def evaluate_compliance_rule_based(record: Dict[str, Any]) -> AudioComplianceReport:
    """
    Evaluates audio metadata and transcript using deterministic Pydantic guardrails.
    Can be seamlessly integrated with an LLM provider (OpenAI, Ollama, etc.) via Instructor.
    
    Args:
        record: Combined record containing DSP, VAD, ASR, and file metadata.
        
    Returns:
        AudioComplianceReport: Validated Pydantic report model instance.
    """
    rejection_reasons = []
    file_name = record.get("file_name", "unknown")
    transcript = record.get("full_transcript", "")
    snr_db = record.get("snr_db", 0.0)
    speech_density = record.get("speech_density_ratio", 0.0)
    confidence = record.get("average_confidence", 0.0)

    # 1. Deterministic Guardrails
    if snr_db < 15.0:
        rejection_reasons.append(f"Low Signal-to-Noise Ratio ({snr_db:.1f} dB < 15.0 dB threshold)")

    if speech_density < 0.40:
        rejection_reasons.append(f"Low Speech Density ({speech_density * 100:.1f}% < 40.0% threshold)")

    if confidence < 0.60:
        rejection_reasons.append(f"Low ASR Acoustic Confidence ({confidence * 100:.1f}% < 60.0% threshold)")

    if not transcript or len(transcript.strip()) == 0:
        rejection_reasons.append("Empty or missing transcript text")

    # 2. Simple Rule-Based PII & Sentiment Mock (Simulating LLM Analysis)
    contains_pii = False
    pii_entities = []
    
    # Simple check example for common keywords
    pii_keywords = ["ssn", "social security", "credit card", "passport"]
    for kw in pii_keywords:
        if kw in transcript.lower():
            contains_pii = True
            pii_entities.append(PIIEntity(entity_type="SENSITIVE_KEYWORD", detected_text=kw))
            rejection_reasons.append(f"Detected sensitive keyword: '{kw}'")

    # 3. Determine Compliance Status Verdict
    if len(rejection_reasons) == 0:
        status = ComplianceStatus.APPROVED
    elif any("Low Signal-to-Noise Ratio" in r or "Low Speech Density" in r for r in rejection_reasons):
        status = ComplianceStatus.QUARANTINED
    else:
        status = ComplianceStatus.REJECTED

    # Build and return validated Pydantic model
    return AudioComplianceReport(
        file_name=file_name,
        contains_pii=contains_pii,
        pii_entities=pii_entities,
        summary=transcript[:150] + ("..." if len(transcript) > 150 else ""),
        sentiment="NEUTRAL",
        audio_clarity_rating=round(float(confidence), 2),
        compliance_status=status,
        rejection_reasons=rejection_reasons
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path for imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.dsp_metrics import analyze_audio_dsp
    from src.vad_engine import process_vad
    from src.asr_diarization import transcribe_audio

    if len(sys.argv) < 2:
        print("Usage: python -m src.compliance_agent <path_to_audio_file>")
        sys.exit(1)

    sample_file = sys.argv[1]
    print(f"Running full pipeline and compliance audit on: {sample_file}\n")

    # Run Modules 1-3
    dsp = analyze_audio_dsp(sample_file)
    vad = process_vad(sample_file)
    asr = transcribe_audio(sample_file, model_size="tiny")

    combined_record = {
        "file_name": Path(sample_file).name,
        "file_path": str(sample_file),
        **dsp,
        **vad,
        **asr
    }

    # Evaluate Compliance Report
    report = evaluate_compliance_rule_based(combined_record)

    print("--- Module 5: Pydantic Compliance Audit Report ---")
    print(report.model_dump_json(indent=2))