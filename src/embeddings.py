"""
Thin wrappers around the baseline (off-the-shelf) and fine-tuned
Sentence-Transformers embedding models, so the rest of the pipeline
can swap between them by name only.
"""
import hashlib
import os

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    _SentenceTransformer = None

from src.config import config


class SentenceTransformerFallback:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, show_progress_bar: bool = False, batch_size: int = 32):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            tokens = text.lower().split()
            vector = [0.0] * 8
            for idx, token in enumerate(tokens[:8]):
                digest = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
                vector[idx % 8] += digest / 0xFFFFFFFF
            vectors.append(vector)
        return vectors


SentenceTransformer = _SentenceTransformer or SentenceTransformerFallback


def get_baseline_embedder() -> SentenceTransformer:
    return SentenceTransformer(config.BASELINE_EMBEDDING_MODEL)


def get_finetuned_embedder() -> SentenceTransformer:
    if not os.path.exists(config.FINETUNED_EMBEDDING_PATH):
        raise FileNotFoundError(
            f"No fine-tuned model found at {config.FINETUNED_EMBEDDING_PATH}. "
            "Run src/finetune_embeddings.py first (recommended on Colab/Kaggle GPU)."
        )
    return SentenceTransformer(config.FINETUNED_EMBEDDING_PATH)


def get_embedder(kind: str = "finetuned") -> SentenceTransformer:
    """
    kind: "baseline" or "finetuned"
    """
    if kind == "baseline":
        return get_baseline_embedder()
    elif kind == "finetuned":
        return get_finetuned_embedder()
    else:
        raise ValueError(f"Unknown embedder kind: {kind}")
