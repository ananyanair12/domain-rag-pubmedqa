"""Retrieval evaluation using PMID ground truth.
Usage:  python -m src.eval_pmid baseline whole [limit]
"""
import json, sys
from pathlib import Path
from src.embeddings import get_embedder
from src.vector_store import get_client, collection_name
from src.evaluate_retrieval import precision_at_k_pmid, hit_at_k, reciprocal_rank

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS = Path(__file__).resolve().parent.parent / "results"

def evaluate(embedder_kind, chunk_size, limit=None, fetch=10):
    rows = [json.loads(l) for l in open(DATA / "eval_questions.jsonl", encoding="utf-8")]
    if limit:
        rows = rows[:limit]

    embedder = get_embedder(embedder_kind)
    qvecs = embedder.encode([r["question"] for r in rows], batch_size=64, show_progress_bar=True)
    qvecs = qvecs.tolist() if hasattr(qvecs, "tolist") else qvecs

    col = get_client().get_collection(collection_name(embedder_kind, chunk_size))
    fetch = max(fetch, 10)
    res = col.query(query_embeddings=qvecs, n_results=fetch)

    ks = (1, 3, 5, 10)
    hit = {kk: [] for kk in ks}
    rrs, precs = [], []
    for r, metas in zip(rows, res["metadatas"]):
        retrieved = [m["pmid"] for m in metas]
        gold = r["pmid"]
        for kk in ks:
            hit[kk].append(hit_at_k(retrieved, gold, kk))
        rrs.append(reciprocal_rank(retrieved[:10], gold))
        precs.append(precision_at_k_pmid(retrieved, gold, 5))

    n = len(rows)
    out = {"embedder": embedder_kind, "chunk_size": str(chunk_size), "n_questions": n}
    for kk in ks:
        out[f"hit@{kk}"] = round(sum(hit[kk]) / n, 4)
    out["mrr@10"] = round(sum(rrs) / n, 4)
    out["precision@5"] = round(sum(precs) / n, 4)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"retrieval_{embedder_kind}_{chunk_size}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(out)
    return out

if __name__ == "__main__":
    kind, size = sys.argv[1], sys.argv[2]
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    evaluate(kind, size, lim)
