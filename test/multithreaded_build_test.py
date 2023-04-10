import numpy

from annoy import AnnoyIndex


def _test_building_with_threads(n_jobs):
    n, f = 10000, 10
    n_trees = 31
    i = AnnoyIndex(f, "euclidean")
    for j in range(n):
        i.add_item(j, numpy.random.normal(size=f))
    assert i.build(n_trees, n_jobs=n_jobs)
    assert n_trees == i.get_n_trees()


def test_one_thread():
    _test_building_with_threads(1)


def test_two_threads():
    _test_building_with_threads(2)


def test_four_threads():
    _test_building_with_threads(4)


def test_eight_threads():
    _test_building_with_threads(8)
