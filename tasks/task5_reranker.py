"""
task5_reranker.py — LLM-powered Re-Ranker
Scores each search result for relevance to the query, then sorts them.
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


class RerankRequest(BaseModel):
    query: str
    results: List[str]


def score_document(query: str, document: str) -> Dict[str, Any]:
    """
    Ask Groq to rate how relevant a document is to the query.
    Returns a dict with score (0.0-1.0) and reason.
    """
    prompt = (
        f"Query: '{query}'\n"
        f"Document: '{document}'\n\n"
        f"Rate how relevant this document is to the query.\n"
        f"Respond ONLY with JSON: "
        f'{{"score": <0.0-1.0>, "reason": "..."}}'
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
    # Fallback
    return {"score": 0.0, "reason": "Could not parse LLM response"}


@router.post("/rerank")
def rerank(body: RerankRequest):
    """
    Step 1: Score each result with Groq.
    Step 2: Sort by score descending.
    Step 3: Return original order alongside re-ranked list.
    """
    original_order = list(body.results)
    scored: List[Dict[str, Any]] = []

    # Score every document
    for doc in body.results:
        result = score_document(body.query, doc)
        scored.append({
            "document": doc,
            "score": float(result.get("score", 0.0)),
            "reason": result.get("reason", ""),
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Attach rank numbers
    reranked = [
        {"rank": i + 1, **item}
        for i, item in enumerate(scored)
    ]

    return {
        "query": body.query,
        "original_order": original_order,
        "reranked_results": reranked,
    }
