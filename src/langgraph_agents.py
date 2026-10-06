"""
src/langgraph_agents.py
-----------------------
Multi-Agent nodes and graph definition using LangGraph.
"""

import duckdb
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.agent_state import AudioAuditState


# --- TOOL DEFINITION ---
def query_duckdb_benchmarks(file_name: str, feature_store_path: str = "data/feature_store/audio_metadata.parquet") -> Dict[str, Any]:
    """Tool used by QA Agent to compare audio metrics against lakehouse averages."""
    try:
        con = duckdb.connect()
        query = f"""
            SELECT 
                AVG(snr_db) as avg_snr,
                AVG(average_confidence) as avg_conf
            FROM read_parquet('{feature_store_path}')
        """
        res = con.execute(query).fetchone()
        return {"lakehouse_avg_snr": res[0], "lakehouse_avg_confidence": res[1]}
    except Exception:
        return {"lakehouse_avg_snr": 20.0, "lakehouse_avg_confidence": 0.80}


# --- NODE 1: Router ---
def quality_router_node(state: AudioAuditState) -> Dict[str, Any]:
    """Routes state based on signal quality heuristics before LLM processing."""
    print(" [Node: Quality Router] Evaluating audio signals...")
    
    # If SNR < 15 dB or confidence < 0.65, route to QA Agent first
    if state["snr_db"] < 15.0 or state["confidence"] < 0.65:
        return {"next_step": "qa_agent"}
    else:
        return {"next_step": "pii_security_agent"}


# --- NODE 2: QA & Diagnostics Agent ---
def qa_agent_node(state: AudioAuditState) -> Dict[str, Any]:
    """Agent that analyzes audio acoustic anomalies using DuckDB metrics."""
    print(" [Node: Audio QA Agent] Investigating poor signal metrics...")
    
    benchmarks = query_duckdb_benchmarks(state["file_name"])
    avg_snr = benchmarks.get("lakehouse_avg_snr", 20.0)
    
    note = (
        f"Signal degradation detected: File SNR is {state['snr_db']:.1f} dB "
        f"(Lakehouse avg: {avg_snr:.1f} dB). Transcript may contain hallucination."
    )
    
    return {
        "qa_notes": note,
        "is_quarantined": True,
        "next_step": "pii_security_agent"
    }


# --- NODE 3: PII & Security Auditor Agent ---
def pii_security_agent_node(state: AudioAuditState) -> Dict[str, Any]:
    """Agent that scans and redacts PII from the raw transcript."""
    print(" [Node: PII Security Agent] Auditing transcript for sensitive entities...")
    
    raw = state["raw_transcript"]
    redacted_entities = []
    sanitized = raw
    
    # Rule/Agentic Redaction Logic (e.g., masking patterns)
    pii_keywords = ["ssn", "social security", "credit card", "passport"]
    for kw in pii_keywords:
        if kw in raw.lower():
            redacted_entities.append(kw.upper())
            sanitized = sanitized.replace(kw, "[REDACTED]")

    return {
        "sanitized_transcript": sanitized,
        "redacted_pii_entities": redacted_entities,
        "next_step": "executive_insight_agent"
    }


# --- NODE 4: Executive Insight Agent ---
def executive_insight_agent_node(state: AudioAuditState) -> Dict[str, Any]:
    """Agent that synthesizes the clean transcript into summaries and action items."""
    print(" [Node: Executive Insight Agent] Generating summary and actionable insights...")
    
    sanitized_text = state.get("sanitized_transcript", "")
    
    summary = f"Executive Summary: {sanitized_text[:120]}..."
    action_items = ["Log processed audio asset into Feature Store", "Notify compliance team if PII was redacted"]

    return {
        "summary": summary,
        "action_items": action_items,
        "next_step": "END"
    }


# --- GRAPH CONSTRUCTION ---
def build_audio_audit_graph():
    workflow = StateGraph(AudioAuditState)
    
    # Add Nodes
    workflow.add_node("router", quality_router_node)
    workflow.add_node("qa_agent", qa_agent_node)
    workflow.add_node("pii_agent", pii_security_agent_node)
    workflow.add_node("insight_agent", executive_insight_agent_node)
    
    # Set Entry Point
    workflow.set_entry_point("router")
    
    # Conditional Edges from Router
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_step"],
        {
            "qa_agent": "qa_agent",
            "pii_security_agent": "pii_agent"
        }
    )
    
    # Linear Transitions
    workflow.add_edge("qa_agent", "pii_agent")
    workflow.add_edge("pii_agent", "insight_agent")
    workflow.add_edge("insight_agent", END)
    
    return workflow.compile()


if __name__ == "__main__":
    app = build_audio_audit_graph()
    
    # Test State Payload
    test_state = {
        "file_name": "2277-149896-0005.flac",
        "raw_transcript": "Many little wrinkles gather between his eyes as he contemplated this.",
        "snr_db": 12.5,  # Trigger QA agent
        "speech_density": 0.85,
        "confidence": 0.75,
        "qa_notes": None,
        "is_quarantined": False,
        "sanitized_transcript": None,
        "redacted_pii_entities": [],
        "summary": None,
        "action_items": [],
        "next_step": "router"
    }
    
    print("--- Running Multi-Agent Audio Audit Graph ---\n")
    final_output = app.invoke(test_state)
    
    print("\n--- Final Graph Execution Result ---")
    print(f"Is Quarantined : {final_output.get('is_quarantined')}")
    print(f"QA Notes       : {final_output.get('qa_notes')}")
    print(f"Sanitized Text : {final_output.get('sanitized_transcript')}")
    print(f"Summary        : {final_output.get('summary')}")