"""
Minimal sanity tests. Run with: pytest tests/
These check the plumbing (chunking, precision@k logic) without
requiring a GPU, API keys, or the full dataset download.
"""
from llama_index.core import Document

from src.chunking import chunk_corpus
from src.evaluate_retrieval import precision_at_k


def test_chunking_produces_nodes():
    docs = [Document(text="Aspirin is a common analgesic. " * 200, doc_id="doc_0")]
    nodes = chunk_corpus(chunk_size=256, docs=docs)
    assert len(nodes) > 1
    assert all(len(n.get_content()) > 0 for n in nodes)


def test_precision_at_k_perfect_match():
    retrieved = ["the exact relevant passage text here", "irrelevant chunk one", "irrelevant chunk two"]
    relevant = "the exact relevant passage text here"
    score = precision_at_k(retrieved, relevant, k=3)
    assert score > 0


def test_precision_at_k_no_match():
    retrieved = ["completely unrelated text", "also unrelated"]
    relevant = "something totally different and unrelated to the above"
    score = precision_at_k(retrieved, relevant, k=2)
    assert score == 0.0
