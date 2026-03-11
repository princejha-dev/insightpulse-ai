"""
agents/pattern_detector.py

Pattern Detection Agent — aggregates all individual analyses
and uses a LangChain chain to detect macro-level patterns
and emerging issues across the feedback batch.
"""

from typing import List, TypedDict
from collections import Counter
import json

from services.gemini_service import get_llm, _clean_json
from utils.prompt import PATTERN_TEMPLATE


class PatternResult(TypedDict):
    total_feedback: int
    sentiment_distribution: dict
    top_complaints: List[dict]
    top_strengths: List[dict]
    urgency_breakdown: dict
    emerging_issues: List[str]


def _count_nested(analyses: List[dict], *keys) -> Counter:
    counts: Counter = Counter()
    for entry in analyses:
        val = entry
        for k in keys:
            val = val.get(k, {}) if isinstance(val, dict) else None
        if val and val != "none":
            counts[str(val)] += 1
    return counts


def _ranked(counter: Counter, total: int, label: str) -> List[dict]:
    return [
        {label: v, "count": c, "percentage": round(c / total * 100, 1) if total else 0}
        for v, c in counter.most_common(10)
    ]


async def detect_patterns(analyses: List[dict]) -> PatternResult:
    """
    Detect patterns across all analyzed feedback entries.

    Uses local counting for speed + a LangChain chain for
    emerging issue detection and nuanced analysis.

    LangChain chain: PATTERN_TEMPLATE | ChatGoogleGenerativeAI

    Args:
        analyses: List of per-feedback analysis dicts.

    Returns:
        PatternResult with aggregated stats and emerging issues.
    """
    total = len(analyses)
    if total == 0:
        return PatternResult(
            total_feedback=0,
            sentiment_distribution={"positive": 0, "negative": 0, "neutral": 0},
            top_complaints=[], top_strengths=[],
            urgency_breakdown={"critical": 0, "high": 0, "medium": 0, "low": 0},
            emerging_issues=[],
        )

    # ── Local counting (no LLM needed for raw stats) ───────────────────────────
    sentiment_c = _count_nested(analyses, "sentiment", "sentiment")
    issue_c     = _count_nested(analyses, "issue", "issue_category")
    strength_c  = _count_nested(analyses, "strength", "strength_category")
    urgency_c   = _count_nested(analyses, "urgency", "urgency")

    sentiment_dist  = {k: sentiment_c.get(k, 0) for k in ("positive", "negative", "neutral")}
    urgency_breakdown = {k: urgency_c.get(k, 0) for k in ("critical", "high", "medium", "low")}
    top_complaints  = _ranked(issue_c, total, "issue")
    top_strengths   = _ranked(strength_c, total, "strength")

    # ── LangChain chain for emerging issues ───────────────────────────────────
    emerging_issues: List[str] = []
    try:
        slim = [
            {
                "feedback": a.get("feedback", ""),
                "sentiment": a.get("sentiment", {}).get("sentiment"),
                "issue": a.get("issue", {}).get("issue_category"),
                "urgency": a.get("urgency", {}).get("urgency"),
                "strength": a.get("strength", {}).get("strength_category"),
            }
            for a in analyses
        ]

        llm = get_llm(temperature=0.2)

        # LangChain LCEL chain: prompt | llm
        chain = PATTERN_TEMPLATE | llm

        response = await chain.ainvoke({"analyses": json.dumps(slim, indent=2)})
        result: dict = json.loads(_clean_json(response.content))
        emerging_issues = result.get("emerging_issues", [])

    except Exception:
        emerging_issues = []

    return PatternResult(
        total_feedback=total,
        sentiment_distribution=sentiment_dist,
        top_complaints=top_complaints,
        top_strengths=top_strengths,
        urgency_breakdown=urgency_breakdown,
        emerging_issues=emerging_issues,
    )