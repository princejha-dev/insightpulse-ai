"""
main.py

AI Customer Feedback Intelligence System
FastAPI Application — Entry Point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from dotenv import load_dotenv

load_dotenv()

from graph.feedback_graph import run_feedback_pipeline
from utils.prompt import load_sample_feedback

app = FastAPI(
    title="AI Customer Feedback Intelligence System",
    description="LangChain + LangGraph multi-agent feedback analysis platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────────

class FeedbackItem(BaseModel):
    message: str
    source: Optional[str] = "unknown"
    date: Optional[str] = None


class FeedbackBatch(BaseModel):
    feedback: List[FeedbackItem]
    business_name: Optional[str] = "Business"
    period: Optional[str] = "Weekly"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "system": "AI Customer Feedback Intelligence System",
        "version": "2.0.0",
        "stack": ["FastAPI", "LangChain", "LangGraph", "Gemini"],
        "status": "running",
    }


@app.post("/analyze-feedback")
async def analyze_feedback(batch: FeedbackBatch):
    """
    Analyze a batch of customer feedback through the
    LangChain + LangGraph multi-agent pipeline.
    """
    if not batch.feedback:
        raise HTTPException(status_code=400, detail="No feedback provided.")

    messages = [item.message for item in batch.feedback]

    try:
        result = await run_feedback_pipeline(
            feedback_messages=messages,
            business_name=batch.business_name,
            period=batch.period,
        )

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/insights.json", "w") as f:
            json.dump(result, f, indent=2)

        return {"status": "success", "report": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/insights")
def get_insights():
    """Return the most recently generated intelligence report."""
    path = "outputs/insights.json"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="No insights yet. Run POST /analyze-feedback first.",
        )
    with open(path) as f:
        return json.load(f)


@app.get("/alerts")
def get_alerts():
    """Return only critical and high-urgency issues."""
    path = "outputs/insights.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="No insights yet.")

    with open(path) as f:
        data = json.load(f)

    alerts = [
        item for item in data.get("individual_analyses", [])
        if item.get("urgency", {}).get("urgency") in ("critical", "high")
    ]
    return {"total_alerts": len(alerts), "alerts": alerts}


@app.post("/analyze-sample")
async def analyze_sample():
    """Run analysis on the built-in sample feedback dataset."""
    messages = load_sample_feedback()
    result = await run_feedback_pipeline(
        feedback_messages=messages,
        business_name="Demo Restaurant",
        period="Weekly",
    )
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/insights.json", "w") as f:
        json.dump(result, f, indent=2)

    return {"status": "success", "report": result}