"""
src/agent_state.py
------------------
Shared state schema passed between LangGraph nodes.
"""

from typing import TypedDict, List, Dict, Any, Optional


class AudioAuditState(TypedDict):
    # Input Data from Feature Store
    file_name: str
    raw_transcript: str
    snr_db: float
    speech_density: float
    confidence: float
    
    # Agent Outputs & Annotations
    qa_notes: Optional[str]
    is_quarantined: bool
    sanitized_transcript: Optional[str]
    redacted_pii_entities: List[str]
    summary: Optional[str]
    action_items: List[str]
    
    # Execution Tracking
    next_step: str