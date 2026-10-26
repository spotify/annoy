import pytest

from annoy import AnnoyIndex


@pytest.mark.parametrize("metric", ["angular", "euclidean", "manhattan", "hamming", "dot"])
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("n_trees", [0, -2])
def test_invalid_tree_count_preserves_index(metric, n_jobs, n_trees):
    index = AnnoyIndex(2, metric)
    index.add_item(0, [1, 0])
    index.add_item(1, [0, 1])

    with pytest.raises(Exception, match="n_trees must be positive or -1"):
        index.build(n_trees, n_jobs=n_jobs)

    assert index.get_n_trees() == 0
    assert index.get_item_vector(0) == [1, 0]
    assert index.get_item_vector(1) == [0, 1]
    assert index.build(1, n_jobs=n_jobs)
    assert index.get_n_trees() == 1
    assert index.get_nns_by_item(0, 1) == [0]


@pytest.mark.parametrize("n_trees", [-1, 1, 3])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_valid_tree_counts(n_trees, n_jobs):
    index = AnnoyIndex(2, "euclidean")
    index.add_item(0, [1, 0])
    index.add_item(1, [0, 1])
    assert index.build(n_trees, n_jobs=n_jobs)
    assert index.get_n_trees() > 0
    if n_trees != -1:
        assert index.get_n_trees() == n_trees
    assert index.get_nns_by_item(0, 2) == [0, 1]


def test_invalid_tree_count_on_disk_can_retry(tmp_path):
    filename = str(tmp_path / "index.ann")
    index = AnnoyIndex(2, "euclidean")
    index.on_disk_build(filename)
    index.add_item(0, [1, 0])
    index.add_item(1, [0, 1])
    with pytest.raises(Exception, match="n_trees must be positive or -1"):
        index.build(0, n_jobs=1)
    assert index.build(1, n_jobs=1)
    index.unload()
    loaded = AnnoyIndex(2, "euclidean")
    assert loaded.load(filename)
    assert loaded.get_nns_by_item(0, 2) == [0, 1]
