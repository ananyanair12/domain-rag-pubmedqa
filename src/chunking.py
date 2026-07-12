"""
Splits the corpus into chunks at several candidate token sizes so a
chunk-size ablation (256 / 512 / 1024) can be run before picking one
size to carry forward into fine-tuning + final indexing.
"""
import json
import uuid

try:
    from llama_index.core import Document as _Document
    from llama_index.core.node_parser import SentenceSplitter as _SentenceSplitter
except Exception:  # pragma: no cover - optional dependency
    _Document = None
    _SentenceSplitter = None

from src.config import config


class Document:
    def __init__(self, text: str, doc_id: str = None, metadata: dict = None):
        self.text = text
        self.doc_id = doc_id
        self.metadata = metadata or {}

    def get_content(self) -> str:
        return self.text


class TextNode:
    def __init__(self, text: str, metadata: dict = None, node_id: str = None):
        self._text = text
        self.metadata = metadata or {}
        self.node_id = node_id or str(uuid.uuid4())

    def get_content(self) -> str:
        return self._text


if _Document is not None and _SentenceSplitter is not None:
    Document = _Document  # type: ignore[misc,assignment]


class SentenceSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)

    def get_nodes_from_documents(self, docs):
        nodes = []
        for doc in docs:
            text = doc.get_content() if hasattr(doc, "get_content") else str(doc)
            if not text:
                continue
            words = text.split()
            step = max(1, self.chunk_size - self.chunk_overlap)
            for idx in range(0, len(words), step):
                chunk_words = words[idx : idx + self.chunk_size]
                if not chunk_words:
                    continue
                nodes.append(
                    TextNode(
                        text=" ".join(chunk_words),
                        metadata=getattr(doc, "metadata", {}) or {},
                        node_id=f"{getattr(doc, 'doc_id', 'doc')}-{len(nodes)}",
                    )
                )
        return nodes


def load_corpus(path: str = None) -> list[Document]:
    path = path or f"{config.DATA_PROCESSED_DIR}/corpus.jsonl"
    docs = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            docs.append(
                Document(
                    text=row["context"],
                    doc_id=f"doc_{i}",
                    metadata={"source_question": row.get("question", "")},
                )
            )
    return docs


def chunk_corpus(chunk_size: int, overlap_ratio: float = None, docs: list[Document] = None):
    """
    Splits documents into nodes (chunks) of ~chunk_size tokens.
    Returns a list of lightweight TextNode objects.
    """
    overlap_ratio = overlap_ratio if overlap_ratio is not None else config.CHUNK_OVERLAP_RATIO
    docs = docs or load_corpus()

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * overlap_ratio),
    )
    nodes = splitter.get_nodes_from_documents(docs)
    return nodes


def build_all_chunk_versions(output_dir: str = None):
    """
    Builds and saves node sets for every chunk size in
    config.CHUNK_SIZES_TO_ABLATE, so each can be embedded and
    evaluated independently (the chunk-size ablation study).
    """
    output_dir = output_dir or config.DATA_PROCESSED_DIR
    docs = load_corpus()

    summary = {}
    for size in config.CHUNK_SIZES_TO_ABLATE:
        nodes = chunk_corpus(chunk_size=size, docs=docs)
        out_path = f"{output_dir}/chunks_{size}.jsonl"
        with open(out_path, "w") as f:
            for n in nodes:
                f.write(
                    json.dumps(
                        {"id": n.node_id, "text": n.get_content(), "metadata": n.metadata}
                    )
                    + "\n"
                )
        summary[size] = len(nodes)
        print(f"Chunk size {size}: {len(nodes)} chunks -> {out_path}")

    return summary


if __name__ == "__main__":
    build_all_chunk_versions()
