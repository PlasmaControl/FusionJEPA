from fusion_jepa.utils.reproducibility import derive_seed, worker_seed


def test_derive_seed_deterministic() -> None:
    assert derive_seed(42, "dataset", 3) == derive_seed(42, "dataset", 3)


def test_derive_seed_sensitive_to_each_component() -> None:
    baseline = derive_seed(42, "dataset", 3)

    assert derive_seed(43, "dataset", 3) != baseline
    assert derive_seed(42, "loader", 3) != baseline
    assert derive_seed(42, "dataset", 4) != baseline
    assert derive_seed(42, 3, "dataset") != baseline


def test_worker_seeds_distinct_across_ranks_and_workers() -> None:
    seeds = {
        worker_seed(42, rank=rank, worker_id=worker_id, epoch=epoch)
        for rank in range(2)
        for worker_id in range(2)
        for epoch in range(2)
    }

    assert len(seeds) == 8
