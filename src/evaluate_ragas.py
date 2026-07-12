"""
Runs the RAGAs evaluation suite (faithfulness, answer relevance,
context precision) over the held-out query set, for both the
baseline and fine-tuned pipelines, so the two can be compared beyond
just retrieval precision.

Note: RAGAs itself calls an LLM (as a judge) to score faithfulness
and relevance. We point RAGAs at Groq via its OpenAI-compatible
wrapper settings; see the `ragas` docs if you swap providers.
"""
import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from src.config import config
from src.evaluate_retrieval import load_eval_queries
from src.pipeline import run_query


def build_ragas_dataset(embedder_kind: str, n_samples: int = 30) -> Dataset:
    """
    Runs n_samples held-out questions through the full pipeline
    (retrieve -> rerank -> generate) and packages the results into the
    format RAGAs expects: question, answer, contexts, ground_truth.
    """
    eval_queries = load_eval_queries()[:n_samples]

    questions, answers, contexts_list, ground_truths = [], [], [], []
    for row in eval_queries:
        result = run_query(row["question"], embedder_kind=embedder_kind, use_reranker=True)
        questions.append(row["question"])
        answers.append(result["answer"])
        contexts_list.append([c["text"] for c in result["retrieved_chunks"]])
        ground_truths.append(row.get("long_answer", row["context"]))

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )


def run_ragas_comparison(n_samples: int = 30):
    results = {}
    for embedder_kind in ["baseline", "finetuned"]:
        print(f"\nRunning RAGAs evaluation for '{embedder_kind}' pipeline "
              f"on {n_samples} held-out queries...")
        dataset = build_ragas_dataset(embedder_kind, n_samples=n_samples)

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        results[embedder_kind] = scores.to_pandas()[
            ["faithfulness", "answer_relevancy", "context_precision"]
        ].mean().to_dict()
        print(f"{embedder_kind}: {results[embedder_kind]}")

    import os
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(f"{config.RESULTS_DIR}/ragas_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== RAGAs Comparison (baseline vs fine-tuned) ===")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_ragas_comparison(n_samples=30)
