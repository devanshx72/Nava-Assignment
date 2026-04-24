"""
main.py — FastAPI entry point.
Loads environment variables, initialises the Groq client, mounts all
task routers, and exposes a health-check endpoint.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Load .env so GROQ_API_KEY is available to all modules at import time
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Initialise Groq client (shared across the app)
# ---------------------------------------------------------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Import task routers AFTER load_dotenv so env vars are already set
# ---------------------------------------------------------------------------
from tasks.task1_rag import router as task1_router
from tasks.task2_agent import router as task2_router
from tasks.task3_llm_judge import router as task3_router
from tasks.task4_hallucination import router as task4_router
from tasks.task5_reranker import router as task5_router
from tasks.task6_guardrails import router as task6_router
from tasks.task7_multiagent import router as task7_router

# ---------------------------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Engineering Showcase",
    description="7-task AI engineering assessment built with FastAPI + Groq",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow all origins for local development
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register all task routers
# ---------------------------------------------------------------------------
app.include_router(task1_router, prefix="/api/task1", tags=["Task 1 - RAG Pipeline"])
app.include_router(task2_router, prefix="/api/task2", tags=["Task 2 - AI Agent"])
app.include_router(task3_router, prefix="/api/task3", tags=["Task 3 - LLM Judge"])
app.include_router(task4_router, prefix="/api/task4", tags=["Task 4 - Hallucination Detector"])
app.include_router(task5_router, prefix="/api/task5", tags=["Task 5 - Re-Ranker"])
app.include_router(task6_router, prefix="/api/task6", tags=["Task 6 - Guardrails"])
app.include_router(task7_router, prefix="/api/task7", tags=["Task 7 - Multi-Agent"])





# ---------------------------------------------------------------------------
# Health-check endpoint
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["Health"])
def health():
    """Simple liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Serve the single-page frontend (Conditional)
# ---------------------------------------------------------------------------
FRONTEND_PATH = os.path.join(os.getcwd(), "frontend")

if os.path.exists(FRONTEND_PATH):
    @app.get("/", tags=["Frontend"])
    def serve_frontend():
        """Serve the single HTML/CSS/JS frontend."""
        return FileResponse("frontend/index.html")

    # Serve static files (CSS, JS) from the frontend directory
    # Mount this last so it doesn't shadow specific routes like "/" or "/api/*"
    app.mount("/", StaticFiles(directory="frontend"), name="frontend")
else:
    @app.get("/", tags=["Frontend"])
    def serve_frontend_missing():
        return {"error": "Frontend directory not found. Please run in terminal mode."}

# ---------------------------------------------------------------------------
# Terminal-based CLI Interface
# ---------------------------------------------------------------------------
def run_terminal_interface():
    """Simple interactive CLI to run the 7 tasks."""
    print("\n" + "="*50)
    print("   AI ENGINEERING SHOWCASE - TERMINAL MODE")
    print("="*50)
    
    while True:
        print("\nAvailable Tasks:")
        print("1. RAG Pipeline")
        print("2. AI Agent with Tool Use")
        print("3. LLM-as-a-Judge")
        print("4. Hallucination Detection")
        print("5. LLM Re-Ranker")
        print("6. Output Guardrails")
        print("7. Multi-Agent System")
        print("0. Exit")
        
        choice = input("\nSelect a task (0-7): ").strip()
        
        if choice == "0":
            break
        
        try:
            if choice == "1":
                from tasks.task1_rag import ingest_documents, query_rag, IngestRequest, QueryRequest
                docs_in = input("Enter documents (comma separated): ")
                docs = [d.strip() for d in docs_in.split(",") if d.strip()]
                if docs:
                    ingest_documents(IngestRequest(documents=docs))
                q = input("Enter your question: ")
                res = query_rag(QueryRequest(query=q))
                print(f"\nAnswer: {res['answer']}")
                
            elif choice == "2":
                from tasks.task2_agent import run_agent, AgentRequest
                q = input("Enter agent query: ")
                res = run_agent(AgentRequest(query=q))
                print(f"\nFinal Answer: {res['final_answer']}")
                
            elif choice == "3":
                from tasks.task3_llm_judge import evaluate_answer, JudgeRequest
                q = input("Question: ")
                a = input("Answer (leave blank for auto): ")
                res = evaluate_answer(JudgeRequest(question=q, answer=a))
                print(f"\nVerdict: {res['verdict']}\nScore: {res['scores']['overall']}")
                
            elif choice == "4":
                from tasks.task4_hallucination import detect_hallucination, HallucinationRequest
                ctx = input("Context: ")
                ans = input("Answer: ")
                res = detect_hallucination(HallucinationRequest(context=ctx, answer=ans))
                print(f"\nVerdict: {res['verdict']}\nScore: {res['hallucination_score']}")
                
            elif choice == "5":
                from tasks.task5_reranker import rerank_results, RerankRequest
                q = input("Query: ")
                res_list = input("Results (comma separated): ").split(",")
                res = rerank_results(RerankRequest(query=q, results=[r.strip() for r in res_list]))
                print("\nRe-ranked Results:")
                for item in res['reranked_results']:
                    print(f"{item['rank']}. {item['document']} (Score: {item['score']})")
                    
            elif choice == "6":
                from tasks.task6_guardrails import check_output, GuardrailRequest
                topic = input("Allowed Topic: ")
                out = input("Output to check: ")
                res = check_output(GuardrailRequest(topic=topic, llm_output=out))
                print(f"\nStatus: {res['overall_status']}")
                
            elif choice == "7":
                from tasks.task7_multiagent import run_pipeline, MultiAgentRequest
                task = input("Research Task: ")
                res = run_pipeline(MultiAgentRequest(task=task))
                print("\nPipeline Output:")
                for step in res['pipeline']:
                    print(f"\n--- {step['agent']} ---\n{step['output']}")
            else:
                print("Invalid choice.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    if os.path.exists(FRONTEND_PATH):
        import uvicorn
        print("Frontend found. Starting GUI server...")
        uvicorn.run(app, host="127.0.0.1", port=8000)
    else:
        run_terminal_interface()
