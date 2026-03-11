"""
agents/sentiment_agent.py

Sentiment Agent — uses a LangChain chain (PromptTemplate | LLM)
to detect positive / negative / neutral sentiment.
"""

from typing import TypedDict
from langchain_core.prompts import PromptTemplate

from services.gemini_service import get_llm, _clean_json
from utils.prompt import SENTIMENT_TEMPLATE
import json


class SentimentResult(TypedDict):
    sentiment: str      # positive | negative | neutral
    confidence: float   # 0.0 – 1.0


async def analyze_sentiment(feedback: str) -> SentimentResult:
    """
    Analyze the sentiment of a single feedback message.

    LangChain chain: SENTIMENT_TEMPLATE | ChatGoogleGenerativeAI

    Args:
        feedback: Raw customer feedback string.

    Returns:
        SentimentResult with 'sentiment' and 'confidence'.
    """
    llm = get_llm(temperature=0.1)

    # LangChain LCEL chain: prompt | llm
    chain = SENTIMENT_TEMPLATE | llm

    try:
        response = await chain.ainvoke({"feedback": feedback})
        raw: str = response.content
        data: dict = json.loads(_clean_json(raw))

        sentiment = data.get("sentiment", "neutral").lower()
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return SentimentResult(sentiment=sentiment, confidence=confidence)

    except Exception:
        return SentimentResult(sentiment="neutral", confidence=0.0)