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

import numpy
import random
from common import TestCase
from annoy import AnnoyIndex


def dot_metric(a, b):
    return -numpy.dot(a, b)


def similarity(a, b):
    # Could replace this with kendall-tau if we're comfortable
    # bringing in scipy as a test dependency.
    return float(len(set(a) & set(b))) / float(len(set(a) | set(b)))


class DotIndexTest(TestCase):
    def test_get_nns_by_vector(self):
        f = 2
        i = AnnoyIndex(f, 'dot')
        i.add_item(0, [2, 2])
        i.add_item(1, [3, 2])
        i.add_item(2, [3, 3])
        i.build(10)

        self.assertEqual(i.get_nns_by_vector([4, 4], 3), [2, 1, 0])
        self.assertEqual(i.get_nns_by_vector([1, 1], 3), [2, 1, 0])
        self.assertEqual(i.get_nns_by_vector([4, 2], 3), [2, 1, 0])

    def test_get_nns_by_item(self):
        f = 2
        i = AnnoyIndex(f, 'dot')
        i.add_item(0, [2, 2])
        i.add_item(1, [3, 2])
        i.add_item(2, [3, 3])
        i.build(10)

        self.assertEqual(i.get_nns_by_item(0, 3), [2, 1, 0])
        self.assertEqual(i.get_nns_by_item(2, 3), [2, 1, 0])

    def test_dist(self):
        f = 2
        i = AnnoyIndex(f, 'dot')
        i.add_item(0, [0, 1])
        i.add_item(1, [1, 1])
        i.add_item(2, [0, 0])

        self.assertAlmostEqual(i.get_distance(0, 1), -1.0)
        self.assertAlmostEqual(i.get_distance(1, 2), 0.0)

    def test_large_index(self):
        f = 1
        i = AnnoyIndex(f, 'dot')
        for j in range(0, 10000):
            i.add_item(j, [j])

        i.build(10)
        for j in range(0, 10000):
            self.assertTrue(i.get_nns_by_item(j, 1)[0] >= j)

    def test_random_accuracy(self):
        n = 10

        n_trees = 10
        n_points = 1000
        n_rounds = 5

        for r in range(n_rounds):
            # create random points at distance x
            f = 10
            idx = AnnoyIndex(f, 'dot')

            data = numpy.array([
                [random.gauss(0, 1) for z in range(f)]
                for j in range(n_points)
            ])

            expected_results = [
                sorted(
                    range(n_points),
                    key=lambda j: dot_metric(data[i], data[j])
                )[:n]
                for i in range(n_points)
            ]

            for i, vec in enumerate(data):
                idx.add_item(i, vec)

            idx.build(n_trees)

            for i in range(n_points):
                nns = idx.get_nns_by_vector(data[i], n)
                self.assertGreater(similarity(nns, expected_results[i]), 0.9)

    def precision(self, n, n_trees=10, n_points=10000, n_rounds=10):
        found = 0
        for r in range(n_rounds):
            # create random points at distance x
            f = 10
            i = AnnoyIndex(f, 'dot')
            for j in range(n_points):
                p = [random.gauss(0, 1) for z in range(f)]
                norm = sum([pi ** 2 for pi in p]) ** 0.5
                x = [pi / norm * j for pi in p]
                i.add_item(j, x)

            i.build(n_trees)

            nns = i.get_nns_by_vector([0] * f, n)
            self.assertEqual(nns, sorted(nns))  # should be in order
            # The number of gaps should be equal to the last item minus n-1
            found += len([y for y in nns if y < n])

        return 1.0 * found / (n * n_rounds)

    def test_precision_10(self):
        self.assertTrue(self.precision(10) >= 0.98)

    def test_precision_100(self):
        self.assertTrue(self.precision(100) >= 0.98)

    def test_precision_1000(self):
        self.assertTrue(self.precision(1000) >= 0.98)

    def test_precision_1000_fewer_trees(self):
        self.assertTrue(self.precision(1000, n_trees=4) >= 0.98)

    def test_get_nns_with_distances(self):
        f = 3
        i = AnnoyIndex(f, 'dot')
        i.add_item(0, [0, 0, 2])
        i.add_item(1, [0, 1, 1])
        i.add_item(2, [1, 0, 0])
        i.build(10)

        l, d = i.get_nns_by_item(0, 3, -1, True)
        self.assertEqual(l, [0, 1, 2])
        self.assertAlmostEqual(d[0], -4.0)
        self.assertAlmostEqual(d[1], -2.0)
        self.assertAlmostEqual(d[2], -0.0)

        l, d = i.get_nns_by_vector([2, 2, 2], 3, -1, True)
        self.assertEqual(l, [0, 1, 2])
        self.assertAlmostEqual(d[0], -4.0)
        self.assertAlmostEqual(d[1], -4.0)
        self.assertAlmostEqual(d[2], -2.0)

    def test_include_dists(self):
        f = 40
        i = AnnoyIndex(f, 'dot')
        v = numpy.random.normal(size=f)
        i.add_item(0, v)
        i.add_item(1, -v)
        i.build(10)

        indices, dists = i.get_nns_by_item(0, 2, 10, True)
        self.assertEqual(indices, [0, 1])
        self.assertAlmostEqual(dists[0], -numpy.dot(v, v))

    def test_distance_consistency(self):
        n, f = 1000, 3
        i = AnnoyIndex(f, 'dot')
        for j in range(n):
            i.add_item(j, numpy.random.normal(size=f))
        i.build(10)
        for a in random.sample(range(n), 100):
            indices, dists = i.get_nns_by_item(a, 100, include_distances=True)
            for b, dist in zip(indices, dists):
                self.assertAlmostEqual(dist, -numpy.dot(
                    i.get_item_vector(a),
                    i.get_item_vector(b)
                ))
                self.assertEqual(dist, i.get_distance(a, b))
