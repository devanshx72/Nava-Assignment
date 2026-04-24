"""
task2_agent.py — AI Agent with Tool Use (ReAct-style loop)
The agent iteratively calls Groq, parses tool-use instructions, executes
the appropriate tool, and loops until it produces a final answer.
"""

import os
import json
import re
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from groq import Groq

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

router = APIRouter()

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# System prompt that explains available tools to the LLM
SYSTEM_PROMPT = """You are a helpful AI assistant with access to two tools.

To use a tool, respond ONLY with valid JSON (no extra text, no markdown):
{"tool": "calculator", "input": {"expression": "3*7"}}
or
{"tool": "web_search", "input": {"query": "latest news about AI"}}

When you have gathered enough information and are ready to give the final answer,
respond ONLY with valid JSON:
{"final_answer": "your complete answer here"}

Rules:
- Always respond with valid JSON — never include extra prose.
- Use the calculator tool for any arithmetic or math.
- Use the web_search tool to look up factual or current information.
- Once you have enough information, output the final_answer immediately.
"""


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class AgentRequest(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def run_calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Only digits, basic operators, dots, spaces and parentheses are allowed.
    """
    # Whitelist: allow only safe characters to prevent code injection
    safe_pattern = re.compile(r"^[\d\s\+\-\*\/\.\(\)\%\^]+$")
    if not safe_pattern.match(expression):
        return "Error: expression contains invalid characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # sandboxed eval
        return str(result)
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


def run_web_search(query: str) -> str:
    """
    Return the top 3 DuckDuckGo search snippets for the query.
    Falls back gracefully if the library is unavailable.
    """
    if not DDGS_AVAILABLE:
        return "Web search is unavailable (duckduckgo-search not installed)."
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(f"- {r.get('title', '')}: {r.get('body', '')}")
        return "\n".join(results) if results else "No results found."
    except Exception as exc:
        return f"Search error: {exc}"


# ---------------------------------------------------------------------------
# POST /run
# ---------------------------------------------------------------------------
@router.post("/run")
def run_agent(body: AgentRequest):
    """
    ReAct-style agent loop:
      1. Send query to Groq with tool-use system prompt.
      2. Parse the JSON response.
      3. If 'tool' key is present, execute the tool and feed result back.
      4. If 'final_answer' key is present, return the answer.
      5. Repeat up to MAX_ITERATIONS times.
    """
    MAX_ITERATIONS = 5

    # Conversation history — starts with the user's query
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": body.query}
    ]

    steps: List[Dict[str, Any]] = []  # audit trail

    for iteration in range(MAX_ITERATIONS):
        # Call Groq with the current conversation history
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()

        # Attempt to parse LLM output as JSON
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting JSON from inside a markdown code block
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    parsed = {"final_answer": raw}
            else:
                parsed = {"final_answer": raw}

        # --- Check for final answer ---
        if "final_answer" in parsed:
            steps.append({"thought": raw, "tool": None, "result": None})
            return {
                "final_answer": parsed["final_answer"],
                "steps": steps,
            }

        # --- Check for tool call ---
        if "tool" in parsed:
            tool_name = parsed.get("tool", "")
            tool_input = parsed.get("input", {})

            # Execute the requested tool
            if tool_name == "calculator":
                tool_result = run_calculator(tool_input.get("expression", ""))
            elif tool_name == "web_search":
                tool_result = run_web_search(tool_input.get("query", ""))
            else:
                tool_result = f"Unknown tool: {tool_name}"

            # Record this step
            steps.append({
                "thought": raw,
                "tool": tool_name,
                "result": tool_result,
            })

            # Append assistant response and tool result to conversation
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}: {tool_result}",
            })
        else:
            # Unexpected format — treat as final answer
            steps.append({"thought": raw, "tool": None, "result": None})
            return {"final_answer": raw, "steps": steps}

    # Safety: max iterations reached
    return {
        "final_answer": "Max iterations reached without a final answer.",
        "steps": steps,
    }
