"""
task1_rag.py — RAG Pipeline
Embeds documents with sentence-transformers, stores them in ChromaDB,
and answers questions by retrieving relevant context then calling Groq.
Supports both plain-text ingestion and PDF file uploads.
"""

import io
import os
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

router = APIRouter()

# ---------------------------------------------------------------------------
# Initialise embedding model (runs on CPU, downloaded once)
# ---------------------------------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# ChromaDB — in-memory client; recreate collection on every server start
# to avoid stale-data errors.
# ---------------------------------------------------------------------------
chroma_client = chromadb.Client()  # ephemeral / in-memory

# Delete collection if it already exists (handles hot-reload)
try:
    chroma_client.delete_collection("rag_collection")
except Exception:
    pass

collection = chroma_client.create_collection("rag_collection")

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    documents: List[str]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------
@router.post("/ingest")
def ingest_documents(body: IngestRequest):
    """
    Embed each document and store it in ChromaDB.
    Uses sentence-transformers 'all-MiniLM-L6-v2' for embeddings.
    """
    global collection

    # Re-create collection to allow re-ingestion without ID conflicts
    try:
        chroma_client.delete_collection("rag_collection")
    except Exception:
        pass
    collection = chroma_client.create_collection("rag_collection")

    documents = body.documents
    # Generate dense embeddings for every document
    embeddings = embedding_model.encode(documents).tolist()

    # Build unique string IDs
    ids = [f"doc_{i}" for i in range(len(documents))]

    # Add to ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
    )

    return {"status": "ingested", "count": len(documents)}


# ---------------------------------------------------------------------------
# POST /ingest_pdf  — upload a PDF file and ingest its pages as chunks
# ---------------------------------------------------------------------------
@router.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF file upload, extract text from every page, embed the pages,
    and store them in ChromaDB.  Each page becomes one chunk.
    """
    global collection

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Read the raw bytes from the uploaded file
    raw_bytes = await file.read()

    # Parse the PDF with pypdf
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {exc}")

    # Extract non-empty pages
    pages = [page.extract_text() for page in reader.pages]
    documents = [text.strip() for text in pages if text and text.strip()]

    if not documents:
        raise HTTPException(
            status_code=422,
            detail="No readable text found in the PDF (it may be scanned/image-only).",
        )

    # Re-create collection to allow re-ingestion without ID conflicts
    try:
        chroma_client.delete_collection("rag_collection")
    except Exception:
        pass
    collection = chroma_client.create_collection("rag_collection")

    # Embed and store
    embeddings = embedding_model.encode(documents).tolist()
    ids = [f"pdf_page_{i}" for i in range(len(documents))]
    collection.add(documents=documents, embeddings=embeddings, ids=ids)

    return {
        "status": "ingested",
        "source": file.filename,
        "pages_ingested": len(documents),
    }


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------
@router.post("/query")
def query_rag(body: QueryRequest):
    """
    Embed the query, retrieve top-k chunks from ChromaDB, then call Groq
    with the retrieved context to produce a grounded answer.
    """
    # Embed the user query
    query_embedding = embedding_model.encode([body.query]).tolist()

    # Retrieve top-k similar documents from ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(body.top_k, collection.count()),
    )

    retrieved_docs = results["documents"][0]   # list of doc strings
    distances = results["distances"][0]        # list of distances

    # Build a grounded prompt that restricts the LLM to the context
    context_text = "\n".join(retrieved_docs)
    prompt = (
        f"Use ONLY the following context to answer.\n"
        f"Context: {context_text}\n"
        f"Question: {body.query}"
    )

    # Call Groq with the grounded prompt
    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_docs,
        "distances": distances,
    }
