import json, random
from pathlib import Path
from datasets import load_dataset

random.seed(42)
OUT = Path(__file__).parent
N_TRAIN = 20000        # homework pairs (kept small so Colab is fast)
N_DISTRACTORS = 10000  # extra abstracts in the library

def text_of(ex):
    return " ".join(ex["context"]["contexts"]).strip()

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    print("Downloading (first time takes a few minutes)...")
    labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    unlabeled = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split="train")

    # EXAM: labeled questions only
    eval_rows = [{
        "pmid": str(ex["pubid"]),
        "question": ex["question"],
        "long_answer": ex.get("long_answer", ""),
        "final_decision": ex.get("final_decision", ""),
    } for ex in labeled]
    exam_ids = {r["pmid"] for r in eval_rows}

    # HOMEWORK: artificial pairs, never overlapping the exam
    art = [ex for ex in artificial if str(ex["pubid"]) not in exam_ids]
    random.shuffle(art)
    train_rows = [{
        "pmid": str(ex["pubid"]),
        "question": ex["question"],
        "passage": text_of(ex),
    } for ex in art[:N_TRAIN]]

    # LIBRARY: exam abstracts + unlabeled distractors
    corpus = {str(ex["pubid"]): text_of(ex) for ex in labeled}
    unl = list(unlabeled)
    random.shuffle(unl)
    for ex in unl[:N_DISTRACTORS]:
        corpus.setdefault(str(ex["pubid"]), text_of(ex))
    corpus_rows = [{"pmid": k, "text": v} for k, v in corpus.items() if v]

    # Safety check: no exam question in homework
    assert not ({r["pmid"] for r in train_rows} & exam_ids), "LEAK!"

    write_jsonl(OUT / "eval_questions.jsonl", eval_rows)
    write_jsonl(OUT / "train_pairs.jsonl", train_rows)
    write_jsonl(OUT / "corpus.jsonl", corpus_rows)
    print("eval:", len(eval_rows), "| train:", len(train_rows), "| corpus:", len(corpus_rows))

if __name__ == "__main__":
    main()