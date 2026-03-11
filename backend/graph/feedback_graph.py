"""
graph/feedback_graph.py

LangGraph Multi-Agent Pipeline Orchestration.

Uses StateGraph from LangGraph to wire all LangChain-powered agents
into a sequential directed pipeline.

Flow:
  Input
    → [Sentiment + Issue + Urgency + Strength Agents]  (per feedback, concurrent)
    → Pattern Detector Agent                            (across all feedback)
    → Insight Generation Agent                          (final recommendations)
    → Assembled Report Output
"""

import asyncio
from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END

from agents.sentiment_agent import analyze_sentiment
from agents.issue_classifier import classify_issue
from agents.urgency_agent import detect_urgency
from agents.strength_agent import detect_strength
from agents.pattern_detector import detect_patterns
from agents.insight_generator import generate_insights


# ── LangGraph State ────────────────────────────────────────────────────────────

class FeedbackState(TypedDict):
    feedback_messages: List[str]
    business_name: str
    period: str
    individual_analyses: List[dict]
    patterns: dict
    insights: dict
    final_report: dict


# ── Node: Per-Feedback Analysis ────────────────────────────────────────────────

async def node_analyze_individual(state: FeedbackState) -> FeedbackState:
    """
    LangGraph Node 1 — Run all four per-feedback LangChain agents
    concurrently for every feedback message.
    """
    async def analyze_one(message: str) -> dict:
        # Run sentiment, issue, and strength concurrently
        sentiment, issue, strength = await asyncio.gather(
            analyze_sentiment(message),
            classify_issue(message),
            detect_strength(message),
        )
        # Urgency depends on the classified issue category
        urgency = await detect_urgency(message, issue["issue_category"])

        return {
            "feedback": message,
            "sentiment": sentiment,
            "issue": issue,
            "urgency": urgency,
            "strength": strength,
        }

    analyses = await asyncio.gather(
        *[analyze_one(msg) for msg in state["feedback_messages"]]
    )
    return {**state, "individual_analyses": list(analyses)}


# ── Node: Pattern Detection ────────────────────────────────────────────────────

async def node_detect_patterns(state: FeedbackState) -> FeedbackState:
    """
    LangGraph Node 2 — Aggregate individual analyses to detect
    macro-level patterns using the LangChain pattern agent.
    """
    patterns = await detect_patterns(state["individual_analyses"])
    return {**state, "patterns": dict(patterns)}


# ── Node: Insight Generation ───────────────────────────────────────────────────

async def node_generate_insights(state: FeedbackState) -> FeedbackState:
    """
    LangGraph Node 3 — Convert patterns into actionable business
    recommendations using the LangChain insight agent.
    """
    insights = await generate_insights(
        patterns=state["patterns"],
        business_name=state["business_name"],
        period=state["period"],
    )
    return {**state, "insights": dict(insights)}


# ── Node: Assemble Report ──────────────────────────────────────────────────────

def node_assemble_report(state: FeedbackState) -> FeedbackState:
    """
    LangGraph Node 4 — Merge all outputs into the final
    intelligence report deliverable.
    """
    p = state["patterns"]
    i = state["insights"]

    report = {
        "business_name":        state["business_name"],
        "period":               state["period"],
        "total_feedback":       p.get("total_feedback", 0),
        "sentiment_distribution": p.get("sentiment_distribution", {}),
        "urgency_breakdown":    p.get("urgency_breakdown", {}),
        "top_complaints":       p.get("top_complaints", []),
        "top_strengths":        p.get("top_strengths", []),
        "emerging_issues":      p.get("emerging_issues", []),
        "executive_summary":    i.get("executive_summary", ""),
        "ai_insights":          i.get("insights", []),
        "strength_insights":    i.get("strength_insights", []),
        "critical_actions":     i.get("critical_actions", []),
        "leverage_actions":     i.get("leverage_actions", []),
        "individual_analyses":  state["individual_analyses"],
    }
    return {**state, "final_report": report}


# ── Graph Assembly ─────────────────────────────────────────────────────────────

def build_feedback_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph pipeline.

    Graph topology (sequential):
      analyze_feedback → detect_patterns → generate_insights → assemble_report → END
    """
    workflow = StateGraph(FeedbackState)

    workflow.add_node("analyze_feedback",  node_analyze_individual)
    workflow.add_node("detect_patterns",   node_detect_patterns)
    workflow.add_node("generate_insights", node_generate_insights)
    workflow.add_node("assemble_report",   node_assemble_report)

    workflow.set_entry_point("analyze_feedback")
    workflow.add_edge("analyze_feedback",  "detect_patterns")
    workflow.add_edge("detect_patterns",   "generate_insights")
    workflow.add_edge("generate_insights", "assemble_report")
    workflow.add_edge("assemble_report",   END)

    return workflow.compile()


# Lazy-loaded singleton graph
_graph = None

def _get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_feedback_graph()
    return _graph


# ── Public API ─────────────────────────────────────────────────────────────────

async def run_feedback_pipeline(
    feedback_messages: List[str],
    business_name: str = "Business",
    period: str = "Weekly",
) -> dict:
    """
    Execute the full LangChain + LangGraph multi-agent pipeline.

    Args:
        feedback_messages: List of raw customer feedback strings.
        business_name:     Business name for context.
        period:            Reporting period label.

    Returns:
        Final assembled intelligence report as a dict.
    """
    graph = _get_graph()

    initial_state = FeedbackState(
        feedback_messages=feedback_messages,
        business_name=business_name,
        period=period,
        individual_analyses=[],
        patterns={},
        insights={},
        final_report={},
    )

    result = await graph.ainvoke(initial_state)
    return result["final_report"]