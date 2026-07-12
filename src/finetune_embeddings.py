"""
Contrastive fine-tuning of a Sentence-Transformers embedding model on
domain (question, relevant-passage) pairs from PubMedQA.

This is the core research contribution of the thesis: does contrastive
fine-tuning of the embedding model improve retrieval quality over the
off-the-shelf baseline?

RECOMMENDED: run this on Google Colab / Kaggle with a free T4 GPU.
Locally on CPU it will still run, just much slower.

Usage:
    python src/finetune_embeddings.py

Output:
    Saves the fine-tuned model to config.FINETUNED_EMBEDDING_PATH
"""
import json
import random

from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader

from src.config import config

SEED = 42


def load_training_pairs(path: str = None) -> list[InputExample]:
    """
    Builds positive (question, relevant-passage) pairs from the training
    pool. MultipleNegativesRankingLoss treats every OTHER pair in the
    batch as an in-batch negative, so we only need positives here.
    """
    path = path or config.TRAIN_PAIRS_PATH
    examples = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            question = row["question"].strip()
            passage = row["context"].strip()
            if question and passage:
                examples.append(InputExample(texts=[question, passage]))
    return examples


def build_hard_negatives(examples: list[InputExample], n_negatives: int = 1):
    """
    Constructs triplets of [anchor, positive, negative] for contrastive learning.
    MultipleNegativesRankingLoss naturally accepts triplets where the third element
    is a hard negative.
    """
    random.seed(SEED)
    passages = [ex.texts[1] for ex in examples]
    augmented = []
    for ex in examples:
        # Simple random sampling for negatives (excluding the correct passage)
        neg = random.choice(passages)
        while neg == ex.texts[1] and len(passages) > 1:
            neg = random.choice(passages)
        augmented.append(InputExample(texts=[ex.texts[0], ex.texts[1], neg]))
    return augmented


def finetune(use_hard_negatives: bool = False):
    print(f"Loading baseline model: {config.BASELINE_EMBEDDING_MODEL}")
    model = SentenceTransformer(config.BASELINE_EMBEDDING_MODEL)

    print("Loading training pairs...")
    examples = load_training_pairs()
    print(f"Loaded {len(examples)} positive (question, passage) pairs")

    if use_hard_negatives:
        examples = build_hard_negatives(examples)
        print(f"After augmenting with negatives: {len(examples)} examples")

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=config.FT_BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * config.FT_EPOCHS * config.FT_WARMUP_RATIO)

    print(
        f"Fine-tuning for {config.FT_EPOCHS} epochs, "
        f"batch size {config.FT_BATCH_SIZE}, lr {config.FT_LEARNING_RATE}..."
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.FT_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.FT_LEARNING_RATE},
        show_progress_bar=True,
    )

    model.save(config.FINETUNED_EMBEDDING_PATH)
    print(f"Fine-tuned model saved to {config.FINETUNED_EMBEDDING_PATH}")


if __name__ == "__main__":
    finetune(use_hard_negatives=False)
