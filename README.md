# AI Engineering Showcase

A full-stack AI engineering assessment app with 7 production-ready AI tasks built with **FastAPI**, **Groq LLM**, **ChromaDB**, and **sentence-transformers**.

> Built as part of the Nava Software internship coding assessment.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (in-memory) |
| Web Search | duckduckgo-search |
| PDF Parsing | pypdf |
| Frontend | Vanilla HTML (`index.html`) + CSS (`style.css`) + JS (`app.js`) |

---

## Setup

### 1. Add your Groq API key

Open `.env` and replace the placeholder:

```
GROQ_API_KEY=your_actual_key_here
```

Get a free key at https://console.groq.com

---

### 2. Activate virtual environment

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: `sentence-transformers` will download the `all-MiniLM-L6-v2` model on first run (~80 MB).

---

### 4. Run the backend

```bash
uvicorn main:app --reload --port 8000
```

---

### 5. Open the frontend

Open your browser at:

```
http://localhost:8000
```

Or open `frontend/index.html` directly in a browser while the server is running on port 8000.

---

## API Endpoints

### Health
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness check |

---

### Task 1 — RAG Pipeline (`/api/task1`)
| Method | Path | Body / Form | Description |
|--------|------|-------------|-------------|
| POST | `/ingest` | `{"documents": [...]}` | Embed and store plain-text documents in ChromaDB |
| POST | `/ingest_pdf` | `multipart/form-data` — `file` field (PDF) | Upload a PDF; each page is extracted and stored as a chunk |
| POST | `/query` | `{"query": "...", "top_k": 3}` | Retrieve context and generate grounded answer |

---

### Task 2 — AI Agent (`/api/task2`)
| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/run` | `{"query": "..."}` | Run ReAct agent with calculator and web search tools |

---

### Task 3 — LLM Judge (`/api/task3`)
| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/evaluate` | `{"question":"...","answer":"...","reference_answer":"..."}` | Score answer on 4 criteria, return PASS/FAIL |

---

### Task 4 — Hallucination Detection (`/api/task4`)
| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/detect` | `{"context":"...","answer":"..."}` | Per-sentence faithfulness check |

---

### Task 5 — Re-Ranker (`/api/task5`)
| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/rerank` | `{"query":"...","results":[...]}` | LLM-scored re-ranking of search results |

---

### Task 6 — Guardrails (`/api/task6`)
| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/check` | `{"topic":"...","llm_output":"..."}` | Off-topic + PII + toxicity checks |

---

### Task 7 — Multi-Agent (`/api/task7`)
| Method | Path | Body | Description |
|--------|------|------|-------------|
| POST | `/run` | `{"task":"..."}` | 3-agent pipeline: Researcher → Analyst → Writer |

---

## Project Structure

```
project/
├── main.py                  # FastAPI entry point
├── tasks/
│   ├── __init__.py
│   ├── task1_rag.py
│   ├── task2_agent.py
│   ├── task3_llm_judge.py
│   ├── task4_hallucination.py
│   ├── task5_reranker.py
│   ├── task6_guardrails.py
│   └── task7_multiagent.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── .env
├── .gitignore               # excludes venv/, .env, __pycache__/
├── requirements.txt
└── README.md
```
