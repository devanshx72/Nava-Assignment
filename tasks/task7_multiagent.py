"""
task7_multiagent.py — Multi-Agent System (3-agent sequential pipeline)
Three specialised agents run in sequence: Researcher -> Analyst -> Writer.
Each agent receives the previous agent's output as its input.
"""

import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from groq import Groq

router = APIRouter()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


class MultiAgentRequest(BaseModel):
    task: str   # research topic or business question


# ---------------------------------------------------------------------------
# Agent definitions — each agent has a name and a system prompt
# ---------------------------------------------------------------------------
AGENTS = [
    {
        "name": "Researcher",
        "system_prompt": (
            "You are a Research Agent. Your job is to gather and summarize "
            "factual information about the given topic. Be thorough and factual. "
            "Output a structured research brief."
        ),
    },
    {
        "name": "Analyst",
        "system_prompt": (
            "You are an Analysis Agent. You receive a research brief and must "
            "analyze it critically. Identify key insights, patterns, risks, and "
            "opportunities. Output a structured analysis report."
        ),
    },
    {
        "name": "Writer",
        "system_prompt": (
            "You are a Writer Agent. You receive an analysis report and must write "
            "a clear, professional final summary suitable for a business audience. "
            "Make it concise, structured, and actionable."
        ),
    },
]


def run_agent(system_prompt: str, user_input: str) -> str:
    """
    Call Groq with the given system prompt and user input.
    Returns the assistant's text response.
    """
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    return response.choices[0].message.content.strip()


@router.post("/run")
def run_pipeline(body: MultiAgentRequest):
    """
    Execute the 3-agent pipeline sequentially.
    Each agent's output becomes the next agent's input.
    Returns the full pipeline trace plus the final report.
    """
    pipeline: List[Dict[str, str]] = []
    current_input = body.task  # first agent receives the raw task

    for agent in AGENTS:
        # Run this agent
        output = run_agent(agent["system_prompt"], current_input)

        # Record the agent's contribution
        pipeline.append({"agent": agent["name"], "output": output})

        # Pass output to next agent
        current_input = output

    # The final agent's output is the deliverable
    final_report = pipeline[-1]["output"]

    return {
        "task": body.task,
        "pipeline": pipeline,
        "final_report": final_report,
    }
