# Copyright (c) 2018 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import random

import numpy
import pytest

from annoy import AnnoyIndex


def dot_metric(a, b):
    return -numpy.dot(a, b)


def recall(retrieved, relevant):
    return float(len(set(relevant) & set(retrieved))) / float(len(set(relevant)))


def test_get_nns_by_vector():
    f = 2
    i = AnnoyIndex(f, "dot")
    i.add_item(0, [2, 2])
    i.add_item(1, [3, 2])
    i.add_item(2, [3, 3])
    i.build(10)

    assert i.get_nns_by_vector([4, 4], 3) == [2, 1, 0]
    assert i.get_nns_by_vector([1, 1], 3) == [2, 1, 0]
    assert i.get_nns_by_vector([4, 2], 3) == [2, 1, 0]


def test_get_nns_by_item():
    f = 2
    i = AnnoyIndex(f, "dot")
    i.add_item(0, [2, 2])
    i.add_item(1, [3, 2])
    i.add_item(2, [3, 3])
    i.build(10)

    assert i.get_nns_by_item(0, 3) == [2, 1, 0]
    assert i.get_nns_by_item(2, 3) == [2, 1, 0]


def test_dist():
    f = 2
    i = AnnoyIndex(f, "dot")
    i.add_item(0, [0, 1])
    i.add_item(1, [1, 1])
    i.add_item(2, [0, 0])
    i.build(10)

    assert i.get_distance(0, 1) == pytest.approx(1.0)
    assert i.get_distance(1, 2) == pytest.approx(0.0)


def recall_at(n, n_trees=10, n_points=1000, n_rounds=5):
    # the best movie/variable name
    total_recall = 0.0

    for r in range(n_rounds):
        # create random points at distance x
        f = 10
        idx = AnnoyIndex(f, "dot")

        data = numpy.array(
            [[random.gauss(0, 1) for z in range(f)] for j in range(n_points)]
        )

        expected_results = [
            sorted(range(n_points), key=lambda j: dot_metric(data[i], data[j]))[:n]
            for i in range(n_points)
        ]

        for i, vec in enumerate(data):
            idx.add_item(i, vec)

        idx.build(n_trees)

        for i in range(n_points):
            nns = idx.get_nns_by_vector(data[i], n)
            total_recall += recall(nns, expected_results[i])

    return total_recall / float(n_rounds * n_points)


def test_recall_at_10():
    value = recall_at(10)
    assert value >= 0.65


def test_recall_at_100():
    value = recall_at(100)
    assert value >= 0.95


def test_recall_at_1000():
    value = recall_at(1000)
    assert value >= 0.99


def test_recall_at_1000_fewer_trees():
    value = recall_at(1000, n_trees=4)
    assert value >= 0.99


def test_get_nns_with_distances():
    f = 3
    i = AnnoyIndex(f, "dot")
    i.add_item(0, [0, 0, 2])
    i.add_item(1, [0, 1, 1])
    i.add_item(2, [1, 0, 0])
    i.build(10)

    l, d = i.get_nns_by_item(0, 3, -1, True)
    assert l == [0, 1, 2]
    assert d[0] == pytest.approx(4)
    assert d[1] == pytest.approx(2)
    assert d[2] == pytest.approx(0)

    l, d = i.get_nns_by_vector([2, 2, 2], 3, -1, True)
    assert l == [0, 1, 2]
    assert d[0] == pytest.approx(4)
    assert d[1] == pytest.approx(4)
    assert d[2] == pytest.approx(2)


def test_include_dists():
    f = 40
    i = AnnoyIndex(f, "dot")
    v = numpy.random.normal(size=f)
    i.add_item(0, v)
    i.add_item(1, -v)
    i.build(10)

    indices, dists = i.get_nns_by_item(0, 2, 10, True)
    assert indices == [0, 1]
    assert dists[0] == pytest.approx(numpy.dot(v, v))


def test_distance_consistency():
    n, f = 1000, 3
    i = AnnoyIndex(f, "dot")
    for j in range(n):
        i.add_item(j, numpy.random.normal(size=f))
    i.build(10)
    for a in random.sample(range(n), 100):
        indices, dists = i.get_nns_by_item(a, 100, include_distances=True)
        for b, dist in zip(indices, dists):
            assert dist == pytest.approx(
                numpy.dot(i.get_item_vector(a), i.get_item_vector(b))
            )
        assert dist == pytest.approx(i.get_distance(a, b))

