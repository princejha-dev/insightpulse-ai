"""
utils/prompt.py

All LangChain PromptTemplate / ChatPromptTemplate definitions
for every agent in the pipeline.
"""

import json
import os
from typing import List

from langchain_core.prompts import PromptTemplate


# ── Sentiment Agent ────────────────────────────────────────────────────────────

SENTIMENT_TEMPLATE = PromptTemplate.from_template("""
You are a sentiment analysis expert. Analyze the following customer feedback.

Feedback: "{feedback}"

Return ONLY valid JSON in exactly this structure:
{{
  "sentiment": "<positive|negative|neutral>",
  "confidence": <float between 0.0 and 1.0>
}}

No explanation. No markdown. JSON only.
""".strip())


# ── Issue Classification Agent ─────────────────────────────────────────────────

ISSUE_TEMPLATE = PromptTemplate.from_template("""
You are a customer feedback classifier.

Feedback: "{feedback}"

Choose ONE category from:
product_quality, service_speed, staff_behavior, pricing, environment,
technical_issue, usability, delivery, payment, other

Return ONLY valid JSON:
{{
  "issue_category": "<category>",
  "description": "<one sentence describing the specific issue>"
}}

No explanation. No markdown. JSON only.
""".strip())


# ── Urgency Detection Agent ────────────────────────────────────────────────────

URGENCY_TEMPLATE = PromptTemplate.from_template("""
You are an urgency assessment expert.

Feedback: "{feedback}"
Issue Category: "{issue_category}"

Levels:
- critical: payment failures, safety issues, system crashes
- high: rude staff, extremely slow service, repeated failures
- medium: moderate inconvenience, fixable issues
- low: minor complaints, preferences, suggestions

Return ONLY valid JSON:
{{
  "urgency": "<critical|high|medium|low>",
  "reason": "<one sentence explaining urgency>"
}}

No explanation. No markdown. JSON only.
""".strip())


# ── Strength Detection Agent ───────────────────────────────────────────────────

STRENGTH_TEMPLATE = PromptTemplate.from_template("""
You are a business strengths analyst.

Feedback: "{feedback}"

Choose the most relevant strength from:
friendly_staff, fast_delivery, product_quality, great_ambience, easy_checkout,
smooth_app_experience, good_value, quick_service, clean_environment,
great_communication, none

Return ONLY valid JSON:
{{
  "strength_category": "<category or none>",
  "description": "<one sentence describing what was praised, empty if none>"
}}

No explanation. No markdown. JSON only.
""".strip())


# ── Pattern Detection Agent ────────────────────────────────────────────────────

PATTERN_TEMPLATE = PromptTemplate.from_template("""
You are a pattern recognition expert analyzing customer feedback data.

Feedback analyses:
{analyses}

Tasks:
1. Count and rank the most frequent issue categories.
2. Count and rank the most frequent strength categories.
3. Identify emerging or critical patterns.
4. Calculate approximate percentages out of total entries.

Return ONLY valid JSON:
{{
  "total_feedback": <int>,
  "sentiment_distribution": {{
    "positive": <int>,
    "negative": <int>,
    "neutral": <int>
  }},
  "top_complaints": [
    {{"issue": "<issue_category>", "count": <int>, "percentage": <float>}}
  ],
  "top_strengths": [
    {{"strength": "<strength_category>", "count": <int>, "percentage": <float>}}
  ],
  "urgency_breakdown": {{
    "critical": <int>, "high": <int>, "medium": <int>, "low": <int>
  }},
  "emerging_issues": ["<issue description>"]
}}

No explanation. No markdown. JSON only.
""".strip())


# ── Insight Generation Agent ───────────────────────────────────────────────────

INSIGHT_TEMPLATE = PromptTemplate.from_template("""
You are a senior business intelligence consultant.

Business: {business_name}
Report Period: {period}

Detected Patterns:
{patterns}

Generate actionable business insights and recommendations.

Return ONLY valid JSON:
{{
  "executive_summary": "<2-3 sentence summary>",
  "insights": [
    {{
      "issue": "<issue name>",
      "impact": "<percentage or count>",
      "recommendation": "<specific actionable recommendation>",
      "priority": "<immediate|short_term|long_term>"
    }}
  ],
  "strength_insights": [
    {{
      "strength": "<strength name>",
      "impact": "<percentage or count>",
      "recommendation": "<how to leverage this strength>"
    }}
  ],
  "critical_actions": ["<action 1>", "<action 2>"],
  "leverage_actions": ["<action 1>", "<action 2>"]
}}

No explanation. No markdown. JSON only.
""".strip())


# ── Prompt Builders (string renderers for agents) ──────────────────────────────

def sentiment_prompt(feedback: str) -> str:
    return SENTIMENT_TEMPLATE.format(feedback=feedback)

def issue_classification_prompt(feedback: str) -> str:
    return ISSUE_TEMPLATE.format(feedback=feedback)

def urgency_prompt(feedback: str, issue_category: str) -> str:
    return URGENCY_TEMPLATE.format(feedback=feedback, issue_category=issue_category)

def strength_detection_prompt(feedback: str) -> str:
    return STRENGTH_TEMPLATE.format(feedback=feedback)

def pattern_detection_prompt(analyses: List[dict]) -> str:
    return PATTERN_TEMPLATE.format(analyses=json.dumps(analyses, indent=2))

def insight_generation_prompt(patterns: dict, business_name: str, period: str) -> str:
    return INSIGHT_TEMPLATE.format(
        patterns=json.dumps(patterns, indent=2),
        business_name=business_name,
        period=period,
    )


# ── Sample Data Loader ─────────────────────────────────────────────────────────

def load_sample_feedback() -> List[str]:
    path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data", "feedback.json")
    )
    if not os.path.exists(path):
        return [
            "The coffee was completely cold when served.",
            "Staff was incredibly friendly and welcoming!",
            "Delivery took over 90 minutes. Unacceptable.",
            "Amazing ambience and beautiful interior design.",
            "The payment terminal crashed twice during checkout.",
            "Waiter was rude and dismissive.",
            "Best burger I have ever had. Will return.",
            "Too noisy during peak hours.",
            "Fast delivery and everything arrived perfectly.",
            "App kept crashing when placing my order.",
        ]
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            return [item.get("message", "") for item in data if item.get("message")]
        return data
    raise ValueError("feedback.json must be a JSON array.")