"""Contrastively fine-tune MiniLM on PubMedQA question-passage pairs.

Run from the project root:
    python -m src.finetune_embeddings --pairs 500 --epochs 1 --out models/smoke_test
"""
import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader, Sampler

from src.config import config

SEED = 42
BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 256
CHECKPOINT_EVERY = 200
LOG_EVERY = 20
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_LOG_PATH = PROJECT_ROOT / "logs" / "finetune.log"


@dataclass(frozen=True)
class TrainingPair:
    pmid: str
    question: str
    passage: str


def load_training_pairs(path=None, limit=None) -> list[TrainingPair]:
    """Load valid PMID/question/passage records, optionally taking the first N."""
    pairs_path = Path(path or config.TRAIN_PAIRS_PATH)
    if limit is not None and limit < 1:
        raise ValueError("--pairs must be a positive integer")

    pairs = []
    with pairs_path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            pmid = str(row["pmid"]).strip()
            question = row["question"].strip()
            passage = row["passage"].strip()
            if pmid and question and passage:
                pairs.append(TrainingPair(pmid, question, passage))
                if limit is not None and len(pairs) == limit:
                    break
    if not pairs:
        raise ValueError(f"No valid training pairs found in {pairs_path}")
    return pairs


def build_pmid_batches(pmids, batch_size: int, seed: int = SEED) -> list[list[int]]:
    """Create shuffled batches that contain at most one pair per PMID."""
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    rng = random.Random(seed)
    grouped_indices = {}
    for index, pmid in enumerate(pmids):
        grouped_indices.setdefault(str(pmid), []).append(index)
    for indices in grouped_indices.values():
        rng.shuffle(indices)

    remaining_pmids = list(grouped_indices)
    rng.shuffle(remaining_pmids)
    batches = []
    while remaining_pmids:
        rng.shuffle(remaining_pmids)
        next_remaining = []
        batch = []
        for pmid in remaining_pmids:
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
            batch.append(grouped_indices[pmid].pop())
            if grouped_indices[pmid]:
                next_remaining.append(pmid)
        if batch:
            batches.append(batch)
        remaining_pmids = next_remaining
    return batches


class PMIDBatchSampler(Sampler):
    def __init__(self, pmids, batch_size: int, seed: int):
        self.pmids = pmids
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        return iter(build_pmid_batches(self.pmids, self.batch_size, self.seed))

    def __len__(self):
        return len(build_pmid_batches(self.pmids, self.batch_size, self.seed))


def _configure_logger() -> logging.Logger:
    TRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("src.finetune_embeddings")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(TRAIN_LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def _latest_checkpoint(checkpoint_root: Path) -> Path:
    checkpoints = [
        path
        for path in checkpoint_root.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No resumable checkpoints found in {checkpoint_root}")
    return max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))


def _checkpoint_state(model, optimizer, scheduler, checkpoint_dir, epoch, next_batch, step, first_loss, last_loss):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint_dir / "model"))
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "next_batch": next_batch,
        "step": step,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, checkpoint_dir / "training_state.pt")


def finetune(
    pairs=None,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    out=config.FINETUNED_EMBEDDING_PATH,
    resume: bool = False,
    seed: int = SEED,
    max_seq_length: int = MAX_SEQ_LENGTH,
    warmup_ratio: float = WARMUP_RATIO,
):
    if pairs is not None and pairs < 1:
        raise ValueError("--pairs must be a positive integer")
    if epochs < 1:
        raise ValueError("--epochs must be a positive integer")
    if batch_size < 1:
        raise ValueError("--batch-size must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("--lr must be positive")
    if max_seq_length < 1:
        raise ValueError("--max-seq-length must be positive")
    if not 0 < warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be between 0 and 1")

    logger = _configure_logger()
    all_pairs = load_training_pairs()
    training_pairs = all_pairs[:pairs] if pairs is not None else all_pairs
    if not training_pairs:
        raise ValueError("The selected training set is empty")
    pmids = [pair.pmid for pair in training_pairs]
    batches_per_epoch = len(PMIDBatchSampler(pmids, batch_size, seed))
    total_steps = batches_per_epoch * epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out_path = Path(out)
    checkpoint_root = out_path / "checkpoints"
    resume_checkpoint = _latest_checkpoint(checkpoint_root) if resume else None
    if resume_checkpoint is None:
        model = SentenceTransformer(BASELINE_MODEL, device=device)
    else:
        model = SentenceTransformer(str(resume_checkpoint / "model"), device=device)
    model.max_seq_length = max_seq_length

    examples = [
        InputExample(texts=[pair.question, pair.passage])
        for pair in training_pairs
    ]
    data_loader = DataLoader(
        examples,
        batch_sampler=PMIDBatchSampler(pmids, batch_size, seed),
        collate_fn=model.smart_batching_collate,
    )
    train_loss = MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def learning_rate_scale(current_step):
        if warmup_steps and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - current_step) / decay_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    start_epoch = 0
    next_batch = 0
    global_step = 0
    first_loss = None
    last_loss = None
    if resume_checkpoint is not None:
        state = torch.load(
            resume_checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        next_batch = state["next_batch"]
        global_step = state["step"]
        first_loss = state["first_loss"]
        last_loss = state["last_loss"]
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        logger.info("Resuming from %s at step %d", resume_checkpoint, global_step)

    logger.info(
        "Training %d/%d pairs on %s for %d epoch(s): batch_size=%d, lr=%g, "
        "warmup_steps=%d, max_seq_length=%d",
        len(training_pairs), len(all_pairs), device, epochs, batch_size,
        learning_rate, warmup_steps, max_seq_length,
    )
    started = time.perf_counter()
    progress_time = started
    progress_step = global_step
    for epoch in range(start_epoch, epochs):
        data_loader.batch_sampler.seed = seed + epoch
        epoch_batches = list(data_loader.batch_sampler)
        model.train()
        for batch_index, (features, labels) in enumerate(data_loader):
            if epoch == start_epoch and batch_index < next_batch:
                continue
            features = [
                {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in feature.items()
                }
                for feature in features
            ]
            labels = labels.to(device)
            loss_value = train_loss(features, labels)
            loss_number = float(loss_value.detach().cpu())
            if first_loss is None:
                first_loss = loss_number
            last_loss = loss_number

            loss_value.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step == 1 or global_step % LOG_EVERY == 0:
                logger.info("step=%d loss=%.6f", global_step, loss_number)
            if global_step % 100 == 0:
                now = time.perf_counter()
                steps_since_report = global_step - progress_step
                interval_seconds_per_step = (now - progress_time) / steps_since_report
                logger.info(
                    "progress step=%d/%d loss=%.6f seconds_per_step=%.3f",
                    global_step, total_steps, loss_number, interval_seconds_per_step,
                )
                progress_time = now
                progress_step = global_step
            if global_step % CHECKPOINT_EVERY == 0:
                checkpoint_epoch = epoch
                checkpoint_batch = batch_index + 1
                if checkpoint_batch == len(epoch_batches):
                    checkpoint_epoch += 1
                    checkpoint_batch = 0
                checkpoint_dir = checkpoint_root / f"checkpoint-{global_step}"
                _checkpoint_state(
                    model, optimizer, scheduler, checkpoint_dir,
                    checkpoint_epoch, checkpoint_batch, global_step,
                    first_loss, last_loss,
                )
                logger.info("Saved checkpoint to %s", checkpoint_dir)

        next_batch = 0

    elapsed = time.perf_counter() - started
    model.save(str(out_path))
    if global_step and global_step % LOG_EVERY:
        logger.info("step=%d loss=%.6f", global_step, last_loss)
    seconds_per_step = elapsed / max(1, global_step - (state["step"] if resume_checkpoint else 0))
    full_steps = len(PMIDBatchSampler([pair.pmid for pair in all_pairs], batch_size, seed)) * epochs
    full_estimate_seconds = seconds_per_step * full_steps
    logger.info("Training loss start=%.6f end=%.6f", first_loss, last_loss)
    logger.info(
        "Training time %.3f seconds/step; estimated full-data runtime %.1f seconds "
        "(%d steps, %d pairs)",
        seconds_per_step, full_estimate_seconds, full_steps, len(all_pairs),
    )
    logger.info("Fine-tuned model saved to %s", out_path)
    return {
        "first_loss": first_loss,
        "last_loss": last_loss,
        "seconds_per_step": seconds_per_step,
        "full_estimate_seconds": full_estimate_seconds,
        "steps": global_step,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=None, help="Number of first training pairs; default: all")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=config.FINETUNED_EMBEDDING_PATH)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint under --out")
    args = parser.parse_args(argv)
    if not 0 < args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be between 0 and 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    finetune(
        pairs=args.pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        out=args.out,
        resume=args.resume,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        warmup_ratio=args.warmup_ratio,
    )


if __name__ == "__main__":
    main()
