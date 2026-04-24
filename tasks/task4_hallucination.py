"""
task4_hallucination.py — Hallucination Detection
Splits an LLM answer into sentences and verifies each one against the
provided source context using Groq, then aggregates a hallucination score.
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


class DetectRequest(BaseModel):
    context: str   # source document
    answer: str    # LLM-generated answer to verify


def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter using punctuation boundaries.
    Filters out very short fragments.
    """
    # Split on period/exclamation/question followed by whitespace or end
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


def verify_sentence(context: str, sentence: str) -> Dict[str, Any]:
    """
    Ask Groq whether the given sentence is SUPPORTED, CONTRADICTED,
    or NOT_MENTIONED given the context.
    Returns a dict with verdict, confidence, and reason.
    """
    prompt = (
        f"Given this source context:\n'{context}'\n\n"
        f"Is the following claim supported, contradicted, or not mentioned?\n"
        f"Claim: '{sentence}'\n\n"
        f"Respond ONLY with JSON:\n"
        f'{{"verdict": "SUPPORTED" or "CONTRADICTED" or "NOT_MENTIONED", '
        f'"confidence": 0.0-1.0, '
        f'"reason": "brief explanation"}}'
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
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
    # Fallback if parsing fails
    return {"verdict": "NOT_MENTIONED", "confidence": 0.5, "reason": raw}


@router.post("/detect")
def detect_hallucination(body: DetectRequest):
    """
    Step 1: Split the answer into sentences.
    Step 2: Verify each sentence against the source context.
    Step 3: Aggregate verdicts into a hallucination score.
    """
    sentences = split_into_sentences(body.answer)

    if not sentences:
        return {
            "hallucination_score": 0.0,
            "verdict": "GROUNDED",
            "sentence_analysis": [],
            "summary": {"supported": 0, "contradicted": 0, "not_mentioned": 0},
        }

    sentence_analysis: List[Dict[str, Any]] = []
    counts = {"SUPPORTED": 0, "CONTRADICTED": 0, "NOT_MENTIONED": 0}

    # Verify each sentence individually
    for sentence in sentences:
        result = verify_sentence(body.context, sentence)
        verdict = result.get("verdict", "NOT_MENTIONED").upper()

        # Normalise verdict to one of the three expected values
        if verdict not in counts:
            verdict = "NOT_MENTIONED"

        counts[verdict] += 1
        sentence_analysis.append({
            "sentence": sentence,
            "verdict": verdict,
            "confidence": result.get("confidence", 0.5),
            "reason": result.get("reason", ""),
        })

    total = len(sentences)
    # Hallucination score: fraction of non-supported sentences
    hallucination_score = (counts["CONTRADICTED"] + counts["NOT_MENTIONED"]) / total
    overall_verdict = "HALLUCINATED" if hallucination_score > 0.3 else "GROUNDED"

    return {
        "hallucination_score": round(hallucination_score, 3),
        "verdict": overall_verdict,
        "sentence_analysis": sentence_analysis,
        "summary": {
            "supported": counts["SUPPORTED"],
            "contradicted": counts["CONTRADICTED"],
            "not_mentioned": counts["NOT_MENTIONED"],
        },
    }
