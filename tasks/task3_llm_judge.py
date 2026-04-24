"""
task3_llm_judge.py — LLM-as-a-Judge Evaluation Pipeline
Optionally generates an answer, then scores it on 4 criteria using Groq.
"""

import os
import json
import re
from fastapi import APIRouter
from pydantic import BaseModel
from groq import Groq

router = APIRouter()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# System prompt for the judge role
JUDGE_SYSTEM_PROMPT = """You are a strict evaluator. Score the answer on these 4 criteria.
Respond ONLY in this JSON format, no extra text:
{
  "accuracy": <1-10>,
  "relevance": <1-10>,
  "completeness": <1-10>,
  "clarity": <1-10>,
  "overall": <1-10>,
  "reasoning": "<one sentence explanation>",
  "verdict": "<PASS or FAIL based on overall >= 7>"
}"""


class EvaluateRequest(BaseModel):
    question: str
    answer: str = ""
    reference_answer: str = ""


@router.post("/evaluate")
def evaluate(body: EvaluateRequest):
    """
    Step 1: If answer is empty, generate one via Groq.
    Step 2: Ask the judge LLM to score the answer on 4 criteria.
    Returns scores, verdict, and reasoning.
    """
    answer = body.answer.strip()

    # --- Step 1: Generate answer if not provided ---
    if not answer:
        gen_response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": body.question}],
        )
        answer = gen_response.choices[0].message.content.strip()

    # --- Step 2: Build the judging prompt ---
    judge_user_prompt = (
        f"Question: {body.question}\n"
        f"Answer: {answer}\n"
    )
    if body.reference_answer:
        judge_user_prompt += f"Reference Answer: {body.reference_answer}\n"

    # Call Groq with the judge system prompt
    judge_response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_user_prompt},
        ],
        temperature=0,
    )
    raw = judge_response.choices[0].message.content.strip()

    # Safely parse JSON from the judge response
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        # Try to pull JSON out of surrounding text
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group())
            except json.JSONDecodeError:
                scores = {"error": "Failed to parse judge response", "raw": raw}
        else:
            scores = {"error": "Failed to parse judge response", "raw": raw}

    return {
        "generated_answer": answer,
        "scores": scores,
        "verdict": scores.get("verdict", "UNKNOWN"),
        "reasoning": scores.get("reasoning", ""),
    }
