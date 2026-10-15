"""
Computes retrieval precision@5 on the held-out evaluation set.

Produces the two headline numbers for the thesis:
  1. Precision@5: baseline embeddings vs. fine-tuned embeddings
     (at the selected chunk size)
  2. Precision@5 across chunk sizes (256 / 512 / 1024) — the
     chunk-size ablation

"Relevant" ground truth: for PubMedQA, we treat the chunk(s) derived
from a question's OWN source context as the relevant passage(s) for
that question. This is a standard weak-labeling approach when a
dataset doesn't ship explicit relevance judgments.
"""
import json

from src.config import config
from src.vector_store import build_collection, query_collection


def load_eval_queries(path: str = None):
    path = path or config.EVAL_QUERIES_PATH
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def check_relevance(retrieved_chunk, target_question: str) -> bool:
    if isinstance(retrieved_chunk, dict):
        metadata = retrieved_chunk.get("metadata") or {}
        source_question = metadata.get("source_question", "")
        if not source_question:
            text = retrieved_chunk.get("text", "")
            source_question = text
        return source_question.strip().lower() == target_question.strip().lower()

    if isinstance(retrieved_chunk, str):
        candidate_text = retrieved_chunk.strip().lower()
        target_text = target_question.strip().lower()
        return target_text == candidate_text or target_text in candidate_text or candidate_text in target_text

    return False


def precision_at_k(retrieved_chunks: list[dict], target_question: str, k: int = 5) -> float:
    top_k = retrieved_chunks[:k]
    hits = sum(1 for c in top_k if check_relevance(c, target_question))
    return hits / min(k, len(top_k)) if top_k else 0.0


def recall_at_k(retrieved_chunks: list[dict], target_question: str, k: int = 5) -> float:
    top_k = retrieved_chunks[:k]
    hits = sum(1 for c in top_k if check_relevance(c, target_question))
    return 1.0 if hits > 0 else 0.0


def mrr_at_k(retrieved_chunks: list[dict], target_question: str, k: int = 5) -> float:
    top_k = retrieved_chunks[:k]
    for idx, c in enumerate(top_k):
        if check_relevance(c, target_question):
            return 1.0 / (idx + 1)
    return 0.0


def evaluate_embedder(embedder_kind: str, chunk_size: int, eval_queries: list[dict], k: int = 5) -> dict:
    p_scores = []
    r_scores = []
    mrr_scores = []
    for row in eval_queries:
        results = query_collection(
            row["question"], embedder_kind=embedder_kind, chunk_size=chunk_size, top_k=k
        )
        p_scores.append(precision_at_k(results, row["question"], k=k))
        r_scores.append(recall_at_k(results, row["question"], k=k))
        mrr_scores.append(mrr_at_k(results, row["question"], k=k))
    return {
        "precision": sum(p_scores) / len(p_scores) if p_scores else 0.0,
        "recall": sum(r_scores) / len(r_scores) if r_scores else 0.0,
        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
    }


def run_before_after_comparison(chunk_size: int = None):
    chunk_size = chunk_size or config.SELECTED_CHUNK_SIZE
    eval_queries = load_eval_queries()

    print(f"Evaluating baseline embeddings @ chunk_size={chunk_size}...")
    baseline_metrics = evaluate_embedder("baseline", chunk_size, eval_queries, k=5)

    print(f"Evaluating fine-tuned embeddings @ chunk_size={chunk_size}...")
    finetuned_metrics = evaluate_embedder("finetuned", chunk_size, eval_queries, k=5)

    print("\n=== Retrieval Evaluation @ K=5: Before vs. After Fine-Tuning ===")
    print(f"Metric      | Baseline | Fine-Tuned | Delta")
    print(f"------------|----------|------------|-------")
    for metric in ["precision", "recall", "mrr"]:
        b_val = baseline_metrics[metric]
        f_val = finetuned_metrics[metric]
        delta = f_val - b_val
        print(f"{metric:<11} | {b_val:.4f}   | {f_val:.4f}     | {delta:+.4f} ({delta / max(b_val, 1e-9) * 100:+.1f}%)")

    result = {
        "chunk_size": chunk_size,
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
        "baseline_precision_at_5": baseline_metrics["precision"],
        "finetuned_precision_at_5": finetuned_metrics["precision"],
        "delta": finetuned_metrics["precision"] - baseline_metrics["precision"],
    }
    _save_result("before_after_finetuning.json", result)
    return result


def run_chunk_size_ablation(embedder_kind: str = "finetuned"):
    eval_queries = load_eval_queries()
    results = {}
    for size in config.CHUNK_SIZES_TO_ABLATE:
        print(f"\nBuilding + evaluating chunk_size={size} with {embedder_kind} embeddings...")
        build_collection(embedder_kind, chunk_size=size)
        metrics = evaluate_embedder(embedder_kind, size, eval_queries, k=5)
        results[str(size)] = metrics["precision"]
        print(f"chunk_size={size} -> precision@5 = {metrics['precision']:.4f}")

    best_size = int(max(results, key=results.get))
    print(f"\nBest chunk size: {best_size} (precision@5 = {results[str(best_size)]:.4f})")

    _save_result("chunk_size_ablation.json", {"results": results, "best_chunk_size": best_size})
    return results


def _save_result(filename: str, data: dict):
    import os
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(f"{config.RESULTS_DIR}/{filename}", "w") as f:
        json.dump(data, f, indent=2)

# ---- PMID-based relevance (Phase B) ----
def precision_at_k_pmid(retrieved_pmids, gold_pmid, k=5):
    """Fraction of the top-k results that come from the gold abstract."""
    top = retrieved_pmids[:k]
    return sum(p == gold_pmid for p in top) / k

def hit_at_k(retrieved_pmids, gold_pmid, k=5):
    """1 if the gold abstract appears anywhere in the top-k, else 0."""
    return float(gold_pmid in retrieved_pmids[:k])

def reciprocal_rank(retrieved_pmids, gold_pmid):
    """1/rank of the first correct result (1.0 = first place, 0 = not found)."""
    for rank, p in enumerate(retrieved_pmids, 1):
        if p == gold_pmid:
            return 1 / rank
    return 0.0
if __name__ == "__main__":
    run_before_after_comparison()