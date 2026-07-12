"""
Builds separate ChromaDB collections for the baseline and fine-tuned
embedding models (at the selected chunk size), so retrieval quality
can be compared side by side using the SAME underlying chunks.
"""
import json

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency
    chromadb = None

from src.config import config
from src.embeddings import get_embedder


class _SimpleCollection:
    def __init__(self, name: str):
        self.name = name
        self._items = []

    def add(self, ids, embeddings, documents, metadatas):
        self._items = [
            {"id": i, "embedding": emb, "document": doc, "metadata": meta}
            for i, emb, doc, meta in zip(ids, embeddings, documents, metadatas)
        ]

    def query(self, query_embeddings, n_results):
        query_vector = query_embeddings[0]
        scored = []
        for item in self._items:
            sim = sum(a * b for a, b in zip(query_vector, item["embedding"]))
            scored.append((sim, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        ranked = scored[:n_results]
        return {
            "documents": [[item["document"] for _, item in ranked]],
            "distances": [[max(0.0, 1.0 - sim) for sim, _ in ranked]],
            "ids": [[item["id"] for _, item in ranked]],
            "metadatas": [[item["metadata"] for _, item in ranked]],
        }


class _SimpleClient:
    def __init__(self, path: str):
        self.path = path
        self._collections = {}

    def delete_collection(self, name: str):
        self._collections.pop(name, None)

    def create_collection(self, name: str):
        self._collections[name] = _SimpleCollection(name)
        return self._collections[name]

    def get_collection(self, name: str):
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' does not exist")
        return self._collections[name]


def get_client():
    if chromadb is None:
        return _SimpleClient(path=config.CHROMA_PERSIST_DIR)
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


def collection_name(embedder_kind: str, chunk_size: int) -> str:
    return f"pubmedqa_{embedder_kind}_{chunk_size}"


def build_collection(embedder_kind: str, chunk_size: int = None):
    """
    embedder_kind: "baseline" or "finetuned"
    Embeds every chunk at the given size with the chosen embedder and
    writes them into a dedicated Chroma collection.
    """
    chunk_size = chunk_size or config.SELECTED_CHUNK_SIZE
    chunks_path = f"{config.DATA_PROCESSED_DIR}/chunks_{chunk_size}.jsonl"

    with open(chunks_path, "r") as f:
        chunks = [json.loads(line) for line in f]

    embedder = get_embedder(embedder_kind)
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [c.get("metadata", {}) for c in chunks]

    print(f"Embedding {len(texts)} chunks with {embedder_kind} model...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    client = get_client()
    name = collection_name(embedder_kind, chunk_size)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name)

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Indexed {len(ids)} chunks into collection '{name}'")
    return name


def query_collection(query: str, embedder_kind: str, chunk_size: int = None, top_k: int = None):
    chunk_size = chunk_size or config.SELECTED_CHUNK_SIZE
    top_k = top_k or config.TOP_K_RETRIEVAL

    client = get_client()
    name = collection_name(embedder_kind, chunk_size)
    collection = client.get_collection(name)

    embedder = get_embedder(embedder_kind)
    query_vec = embedder.encode([query])
    if hasattr(query_vec, "tolist"):
        query_vec = query_vec.tolist()

    results = collection.query(query_embeddings=query_vec, n_results=top_k)
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]
    metadatas = results.get("metadatas", [None])[0] if results.get("metadatas") else [None] * len(ids)

    return [
        {"id": i, "text": d, "distance": dist, "metadata": meta}
        for i, d, dist, meta in zip(ids, docs, distances, metadatas)
    ]


if __name__ == "__main__":
    build_collection("baseline", config.SELECTED_CHUNK_SIZE)
    build_collection("finetuned", config.SELECTED_CHUNK_SIZE)
