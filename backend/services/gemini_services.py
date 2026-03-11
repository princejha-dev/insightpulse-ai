"""
services/gemini_service.py

LangChain-based Gemini wrapper.
Uses ChatGoogleGenerativeAI from langchain-google-genai.
All agents call this service — never the raw Gemini SDK directly.
"""

import os
import json
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Return a configured LangChain ChatGoogleGenerativeAI instance.

    Args:
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        ChatGoogleGenerativeAI LLM instance.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Please export GEMINI_API_KEY=<your-key> before running."
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
    )


def _clean_json(raw: str) -> str:
    """Strip markdown code fences that Gemini sometimes wraps around JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


async def invoke_llm(prompt: str, expect_json: bool = True) -> Any:
    """
    Invoke the LangChain LLM with a plain string prompt.

    Args:
        prompt:      Full prompt string.
        expect_json: If True, parse response as JSON dict/list.

    Returns:
        Parsed dict/list (if expect_json=True) or raw string.
    """
    llm = get_llm()

    # Build a simple ChatPromptTemplate and chain it with the LLM
    prompt_template = ChatPromptTemplate.from_messages(
        [("human", "{input}")]
    )
    chain = prompt_template | llm

    try:
        response = await chain.ainvoke({"input": prompt})
        raw_text: str = response.content
    except Exception as exc:
        raise RuntimeError(f"LangChain/Gemini invocation error: {exc}") from exc

    if not expect_json:
        return raw_text

    cleaned = _clean_json(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-JSON output.\n"
            f"Raw: {raw_text}\n"
            f"Error: {exc}"
        ) from exc