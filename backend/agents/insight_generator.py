"""
agents/insight_generator.py

Insight Generation Agent — uses a LangChain chain to convert
detected patterns into actionable business recommendations.
"""

from typing import List, TypedDict
import json

from services.gemini_service import get_llm, _clean_json
from utils.prompt import INSIGHT_TEMPLATE


class InsightItem(TypedDict):
    issue: str
    impact: str
    recommendation: str
    priority: str           # immediate | short_term | long_term


class StrengthInsightItem(TypedDict):
    strength: str
    impact: str
    recommendation: str


class InsightReport(TypedDict):
    executive_summary: str
    insights: List[InsightItem]
    strength_insights: List[StrengthInsightItem]
    critical_actions: List[str]
    leverage_actions: List[str]


async def generate_insights(
    patterns: dict,
    business_name: str = "Business",
    period: str = "Weekly",
) -> InsightReport:
    """
    Generate actionable business intelligence from detected patterns.

    LangChain chain: INSIGHT_TEMPLATE | ChatGoogleGenerativeAI

    Args:
        patterns:      PatternResult dict from Pattern Detection Agent.
        business_name: Name of the business for context.
        period:        Reporting period (e.g., "Weekly", "Monthly").

    Returns:
        InsightReport with prioritized recommendations and strategic actions.
    """
    llm = get_llm(temperature=0.3)

    # LangChain LCEL chain: prompt | llm
    chain = INSIGHT_TEMPLATE | llm

    try:
        response = await chain.ainvoke({
            "patterns": json.dumps(patterns, indent=2),
            "business_name": business_name,
            "period": period,
        })
        result: dict = json.loads(_clean_json(response.content))

        return InsightReport(
            executive_summary=result.get("executive_summary", ""),
            insights=result.get("insights", []),
            strength_insights=result.get("strength_insights", []),
            critical_actions=result.get("critical_actions", []),
            leverage_actions=result.get("leverage_actions", []),
        )

    except Exception:
        return InsightReport(
            executive_summary="Insight generation encountered an error.",
            insights=[], strength_insights=[],
            critical_actions=[], leverage_actions=[],
        )