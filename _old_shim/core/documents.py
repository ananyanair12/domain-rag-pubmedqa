class Document:
    def __init__(self, text: str, doc_id: str | None = None, metadata: dict | None = None):
        self.text = text
        self.doc_id = doc_id
        self.metadata = metadata or {}

    def get_content(self) -> str:
        return self.text
