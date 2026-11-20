from src.evaluate_retrieval import hit_at_k, precision_at_k_pmid, reciprocal_rank


def test_hit_at_k_checks_only_the_top_k():
    retrieved = ["other", "gold", "gold"]

    assert hit_at_k(retrieved, "gold", k=1) == 0.0
    assert hit_at_k(retrieved, "gold", k=2) == 1.0


def test_reciprocal_rank_uses_first_matching_result():
    assert reciprocal_rank(["other", "gold", "gold"], "gold") == 0.5
    assert reciprocal_rank(["other"], "gold") == 0.0


def test_precision_at_k_pmid_counts_relevant_results_in_top_k():
    assert precision_at_k_pmid(["gold", "other", "gold", "gold"], "gold", k=3) == 2 / 3
    assert precision_at_k_pmid(["other", "gold"], "gold", k=2) == 0.5
