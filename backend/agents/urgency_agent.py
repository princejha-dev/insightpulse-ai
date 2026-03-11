"""
agents/urgency_agent.py

Urgency Detection Agent — uses a LangChain chain to score
the severity of a complaint: critical | high | medium | low.
"""

from typing import TypedDict
import json

from services.gemini_service import get_llm, _clean_json
from utils.prompt import URGENCY_TEMPLATE


URGENCY_LEVELS = ("critical", "high", "medium", "low")

CRITICAL_KEYWORDS = {
    "payment failed", "crash", "system down", "safety",
    "injury", "fraud", "data breach", "unauthorized charge",
}
HIGH_KEYWORDS = {
    "rude", "disrespectful", "extremely slow", "waited forever",
    "never arrived", "missing order",
}


class UrgencyResult(TypedDict):
    urgency: str
    reason: str


def _keyword_escalate(feedback: str) -> str | None:
    lower = feedback.lower()
    for kw in CRITICAL_KEYWORDS:
        if kw in lower:
            return "critical"
    for kw in HIGH_KEYWORDS:
        if kw in lower:
            return "high"
    return None


async def detect_urgency(feedback: str, issue_category: str) -> UrgencyResult:
    """
    Determine the urgency level of an issue.

    LangChain chain: URGENCY_TEMPLATE | ChatGoogleGenerativeAI

    Args:
        feedback:        Raw customer feedback string.
        issue_category:  Category from the Issue Classifier agent.

    Returns:
        UrgencyResult with 'urgency' and 'reason'.
    """
    fast = _keyword_escalate(feedback)
    if fast == "critical":
        return UrgencyResult(
            urgency="critical",
            reason="Critical keyword detected (payment/safety/crash).",
        )

    llm = get_llm(temperature=0.1)

    # LangChain LCEL chain: prompt | llm
    chain = URGENCY_TEMPLATE | llm

    try:
        response = await chain.ainvoke({
            "feedback": feedback,
            "issue_category": issue_category,
        })
        data: dict = json.loads(_clean_json(response.content))

        urgency = data.get("urgency", "low").lower()
        if urgency not in URGENCY_LEVELS:
            urgency = "medium"

        # Honour keyword fast-path escalation
        if fast == "high" and URGENCY_LEVELS.index(urgency) > URGENCY_LEVELS.index("high"):
            urgency = "high"

        return UrgencyResult(
            urgency=urgency,
            reason=data.get("reason", "No reason provided."),
        )

    except Exception:
        return UrgencyResult(urgency="low", reason="Could not determine urgency.")