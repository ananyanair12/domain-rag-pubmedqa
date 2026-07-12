"""
Cross-encoder reranking layer applied on top of the vector-search
candidates, to see how much additional ranking quality it adds over
raw vector similarity.
"""
try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
except Exception:  # pragma: no cover - optional dependency
    _CrossEncoder = None

from src.config import config

_reranker = None


class CrossEncoderFallback:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_terms = set(query.lower().split())
            text_terms = set(text.lower().split())
            overlap = len(query_terms & text_terms)
            scores.append(float(overlap))
        return scores


def get_reranker():
    global _reranker
    if _reranker is None:
        if _CrossEncoder is None:
            _reranker = CrossEncoderFallback(config.RERANKER_MODEL)
        else:
            _reranker = _CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = None) -> list[dict]:
    """
    candidates: list of {"id", "text", "distance"} from vector_store.query_collection
    Returns the same dicts, re-sorted by cross-encoder relevance score,
    truncated to top_k.
    """
    top_k = top_k or config.TOP_K_RERANKED
    reranker = get_reranker()

    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]
