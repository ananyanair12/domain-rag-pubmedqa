"""
Downloads PubMedQA (pqa_labeled subset) via HuggingFace `datasets`,
and splits it into:
  - a training pool used to build contrastive fine-tuning pairs
  - a held-out evaluation pool used ONLY for retrieval / RAGAs evaluation

Run this once, locally or on Colab:
    python data/download_pubmedqa.py

Swap the `load_dataset(...)` call for "financial_phrasebank"/FiQA or CUAD
if you choose the fintech or legal variant instead of biomedical.
"""
import json
import os
import random

from datasets import load_dataset

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"
EVAL_HOLD_OUT_RATIO = 0.15  # 15% of Q/A pairs reserved strictly for evaluation
SEED = 42


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Downloading PubMedQA (pqa_labeled)...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    ds.to_json(f"{RAW_DIR}/pubmedqa_labeled_raw.jsonl")

    records = []
    for row in ds:
        # Each row: question, context (list of sentences), long_answer, final_decision
        context_text = " ".join(row["context"]["contexts"]) if row.get("context") else ""
        if not context_text or not row.get("question"):
            continue
        records.append(
            {
                "question": row["question"],
                "context": context_text,
                "long_answer": row.get("long_answer", ""),
                "label": row.get("final_decision", ""),
            }
        )

    random.seed(SEED)
    random.shuffle(records)

    n_eval = int(len(records) * EVAL_HOLD_OUT_RATIO)
    eval_records = records[:n_eval]
    train_records = records[n_eval:]

    with open(f"{PROCESSED_DIR}/train_pairs.jsonl", "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(f"{PROCESSED_DIR}/eval_queries.jsonl", "w") as f:
        for r in eval_records:
            f.write(json.dumps(r) + "\n")

    # The full corpus (all contexts) is what gets chunked + indexed.
    with open(f"{PROCESSED_DIR}/corpus.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps({"question": r["question"], "context": r["context"]}) + "\n")

    print(f"Total records: {len(records)}")
    print(f"Train pool (for fine-tuning): {len(train_records)}")
    print(f"Held-out eval pool (untouched during fine-tuning): {len(eval_records)}")
    print(f"Saved to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
