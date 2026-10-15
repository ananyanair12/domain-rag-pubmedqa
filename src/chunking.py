import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SIZES = {"128": 128, "256": 256, "512": 512, "whole": None}

def chunk_words(text, size, overlap_ratio=0.15):
    words = text.split()
    if size is None or len(words) <= size:
        return [text]
    step = max(1, int(size * (1 - overlap_ratio)))
    chunks = []
    for i in range(0, len(words), step):
        piece = words[i:i + size]
        if piece:
            chunks.append(" ".join(piece))
        if i + size >= len(words):
            break
    return chunks

def build_chunks():
    corpus = [json.loads(l) for l in open(DATA / "corpus.jsonl", encoding="utf-8")]
    for label, size in SIZES.items():
        out = DATA / f"chunks_{label}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for doc in corpus:
                for i, c in enumerate(chunk_words(doc["text"], size)):
                    f.write(json.dumps({
                        "chunk_id": f'{doc["pmid"]}_{i}',
                        "pmid": doc["pmid"],
                        "text": c,
                    }, ensure_ascii=False) + "\n")
        print("wrote", out.name)

if __name__ == "__main__":
    build_chunks()