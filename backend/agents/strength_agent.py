"""
agents/strength_agent.py

Strength Detection Agent — uses a LangChain chain to surface
business strengths from positive customer feedback.
"""

from typing import TypedDict
import json

from services.gemini_service import get_llm, _clean_json
from utils.prompt import STRENGTH_TEMPLATE


VALID_STRENGTHS = {
    "friendly_staff", "fast_delivery", "product_quality", "great_ambience",
    "easy_checkout", "smooth_app_experience", "good_value", "quick_service",
    "clean_environment", "great_communication", "none",
}


class StrengthResult(TypedDict):
    strength_category: str
    description: str


async def detect_strength(feedback: str) -> StrengthResult:
    """
    Identify what a customer is praising in their feedback.

    LangChain chain: STRENGTH_TEMPLATE | ChatGoogleGenerativeAI

    Args:
        feedback: Raw customer feedback string.

    Returns:
        StrengthResult with 'strength_category' and 'description'.
    """
    llm = get_llm(temperature=0.1)

    # LangChain LCEL chain: prompt | llm
    chain = STRENGTH_TEMPLATE | llm

    try:
        response = await chain.ainvoke({"feedback": feedback})
        data: dict = json.loads(_clean_json(response.content))

        category = data.get("strength_category", "none").lower()
        if category not in VALID_STRENGTHS:
            category = "none"

        return StrengthResult(
            strength_category=category,
            description=data.get("description", ""),
        )

    except Exception:
        return StrengthResult(strength_category="none", description="Could not detect strength.")