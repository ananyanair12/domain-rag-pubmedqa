import json

import pytest

from src.finetune_embeddings import build_pmid_batches, load_training_pairs


def test_load_training_pairs_reads_passage_and_applies_limit(tmp_path):
    pair_file = tmp_path / "pairs.jsonl"
    rows = [
        {"pmid": 101, "question": " question ", "passage": " passage "},
        {"pmid": 102, "question": "next", "passage": "abstract"},
    ]
    pair_file.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    pairs = load_training_pairs(pair_file, limit=1)

    assert len(pairs) == 1
    assert (pairs[0].pmid, pairs[0].question, pairs[0].passage) == (
        "101",
        "question",
        "passage",
    )


def test_pmid_batches_cover_each_pair_without_duplicate_pmids():
    pmids = ["a", "a", "b", "c", "c", "d", "a"]

    batches = build_pmid_batches(pmids, batch_size=3, seed=7)

    assert sorted(index for batch in batches for index in batch) == list(range(len(pmids)))
    assert all(len(batch) <= 3 for batch in batches)
    assert all(
        len({pmids[index] for index in batch}) == len(batch)
        for batch in batches
    )


def test_load_training_pairs_rejects_nonpositive_limit(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        load_training_pairs(tmp_path / "unused.jsonl", limit=0)
