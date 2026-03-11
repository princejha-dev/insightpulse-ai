"""
agents/issue_classifier.py

Issue Classification Agent — uses a LangChain chain
(PromptTemplate | LLM) to classify the problem type in feedback.
"""

from typing import TypedDict
import json

from services.gemini_service import get_llm, _clean_json
from utils.prompt import ISSUE_TEMPLATE


VALID_CATEGORIES = {
    "product_quality", "service_speed", "staff_behavior", "pricing",
    "environment", "technical_issue", "usability", "delivery", "payment", "other",
}


class IssueResult(TypedDict):
    issue_category: str
    description: str


async def classify_issue(feedback: str) -> IssueResult:
    """
    Classify the primary issue in a feedback message.

    LangChain chain: ISSUE_TEMPLATE | ChatGoogleGenerativeAI

    Args:
        feedback: Raw customer feedback string.

    Returns:
        IssueResult with 'issue_category' and 'description'.
    """
    llm = get_llm(temperature=0.1)

    # LangChain LCEL chain: prompt | llm
    chain = ISSUE_TEMPLATE | llm

    try:
        response = await chain.ainvoke({"feedback": feedback})
        data: dict = json.loads(_clean_json(response.content))

        category = data.get("issue_category", "other").lower()
        if category not in VALID_CATEGORIES:
            category = "other"

        return IssueResult(
            issue_category=category,
            description=data.get("description", "No description."),
        )

    except Exception:
        return IssueResult(issue_category="other", description="Could not classify.")