class SentenceSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)

    def get_nodes_from_documents(self, docs):
        nodes = []
        for doc in docs:
            text = doc.get_content() if hasattr(doc, "get_content") else str(doc)
            words = text.split()
            step = max(1, self.chunk_size - self.chunk_overlap)
            for idx in range(0, len(words), step):
                chunk_words = words[idx : idx + self.chunk_size]
                if chunk_words:
                    nodes.append(type("TextNode", (), {"_text": " ".join(chunk_words), "metadata": getattr(doc, "metadata", {}) or {}, "node_id": getattr(doc, "doc_id", "doc"), "get_content": lambda self: self._text})())
        return nodes
