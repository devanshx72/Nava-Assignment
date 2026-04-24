"""
task6_guardrails.py — LLM Output Guardrails
Runs three independent checks (off-topic, PII, toxicity) and aggregates
the results into an overall ALLOWED / BLOCKED verdict.
"""

import os
import json
import re
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from groq import Groq

router = APIRouter()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


class GuardrailRequest(BaseModel):
    topic: str        # the allowed / expected topic
    llm_output: str   # text to check


# ---------------------------------------------------------------------------
# CHECK 1 — Off-topic detection (LLM-based)
# ---------------------------------------------------------------------------
def check_off_topic(topic: str, llm_output: str) -> Dict[str, Any]:
    """Ask Groq whether the output is on-topic or off-topic."""
    prompt = (
        f"The allowed topic is: '{topic}'.\n"
        f"Is the following response on-topic or off-topic?\n"
        f"Response: '{llm_output}'\n"
        f"Reply ONLY with JSON: "
        f'{{"verdict": "ON_TOPIC" or "OFF_TOPIC", "confidence": 0.0-1.0, "reason": "..."}}'
    )
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"verdict": "OFF_TOPIC", "confidence": 0.5, "reason": raw}


# ---------------------------------------------------------------------------
# CHECK 2 — PII detection (rule-based regex, no LLM)
# ---------------------------------------------------------------------------
PII_PATTERNS: Dict[str, str] = {
    "email":       r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone":       r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
    "aadhaar":     r"\b\d{4}\s\d{4}\s\d{4}\b",
}


def check_pii(llm_output: str) -> Dict[str, Any]:
    """Detect PII using regex patterns and return redacted text."""
    found_types: List[str] = []
    redacted = llm_output

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, llm_output)
        if matches:
            found_types.append(pii_type)
            # Replace all occurrences with [REDACTED]
            redacted = re.sub(pattern, "[REDACTED]", redacted)

    return {
        "found": len(found_types) > 0,
        "types": found_types,
        "redacted_text": redacted,
    }


# ---------------------------------------------------------------------------
# CHECK 3 — Toxicity detection (LLM-based)
# ---------------------------------------------------------------------------
def check_toxicity(llm_output: str) -> Dict[str, Any]:
    """Ask Groq whether the text contains toxic or harmful content."""
    prompt = (
        f"Does the following text contain toxic, harmful, or inappropriate content?\n"
        f"Text: '{llm_output}'\n"
        f"Reply ONLY with JSON: "
        f'{{"is_toxic": true/false, "confidence": 0.0-1.0, "reason": "..."}}'
    )
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"is_toxic": False, "confidence": 0.5, "reason": raw}


# ---------------------------------------------------------------------------
# POST /check
# ---------------------------------------------------------------------------
@router.post("/check")
def check_output(body: GuardrailRequest):
    """
    Run all three guardrail checks (off-topic, PII, toxicity) and
    return an aggregated ALLOWED / BLOCKED decision.
    """
    # Run checks sequentially (can be parallelised with asyncio.gather if needed)
    off_topic_result = check_off_topic(body.topic, body.llm_output)
    pii_result = check_pii(body.llm_output)
    toxicity_result = check_toxicity(body.llm_output)

    # Determine if any check should block the output
    is_blocked = (
        off_topic_result.get("verdict") == "OFF_TOPIC"
        or pii_result.get("found", False)
        or toxicity_result.get("is_toxic", False)
    )

    return {
        "overall_status": "BLOCKED" if is_blocked else "ALLOWED",
        "checks": {
            "off_topic": off_topic_result,
            "pii": pii_result,
            "toxicity": toxicity_result,
        },
    }
