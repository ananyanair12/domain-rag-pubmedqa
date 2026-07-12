"""
End-to-end RAG pipeline: query -> retrieve -> rerank -> generate.
Used by the FastAPI backend, the Streamlit dashboard, and the
evaluation scripts, so all three stay consistent.
"""
import time

from src.config import config
from src.generation import generate_answer
from src.reranker import rerank
from src.vector_store import query_collection


def run_query(question: str, embedder_kind: str = "finetuned", use_reranker: bool = True) -> dict:
    """
    embedder_kind: "baseline" or "finetuned" — lets us run the exact
    same question through both pipelines for comparison.
    """
    start = time.time()

    candidates = query_collection(
        question, embedder_kind=embedder_kind, top_k=config.TOP_K_RETRIEVAL
    )

    if use_reranker:
        final_chunks = rerank(question, candidates, top_k=config.TOP_K_RERANKED)
    else:
        final_chunks = candidates[: config.TOP_K_RERANKED]

    contexts = [c["text"] for c in final_chunks]
    answer = generate_answer(question, contexts)

    latency = time.time() - start

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": final_chunks,
        "embedder_used": embedder_kind,
        "reranked": use_reranker,
        "latency_seconds": round(latency, 3),
    }
