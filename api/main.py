"""
FastAPI backend for the RAG pipeline.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /query      -> ask a question, get answer + retrieved chunks + metrics
    GET  /health      -> basic health check
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline import run_query

app = FastAPI(
    title="Domain-Specific RAG API",
    description="RAG system with contrastively fine-tuned embeddings (PubMedQA).",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str
    embedder: str = "finetuned"   # "finetuned" or "baseline"
    use_reranker: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list
    embedder_used: str
    reranked: bool
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if req.embedder not in ("baseline", "finetuned"):
        raise HTTPException(status_code=400, detail="embedder must be 'baseline' or 'finetuned'")
    try:
        result = run_query(req.question, embedder_kind=req.embedder, use_reranker=req.use_reranker)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
