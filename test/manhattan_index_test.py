# Copyright (c) 2013 Spotify AB
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


class ManhattanIndexTest(TestCase):
    def test_get_nns_by_vector(self):
        f = 2
        i = AnnoyIndex(f, 'manhattan')
        i.add_item(0, [2, 2])
        i.add_item(1, [3, 2])
        i.add_item(2, [3, 3])
        i.build(10)

        self.assertEqual(i.get_nns_by_vector([4, 4], 3), [2, 1, 0])
        self.assertEqual(i.get_nns_by_vector([1, 1], 3), [0, 1, 2])
        self.assertEqual(i.get_nns_by_vector([5, 3], 3), [2, 1, 0])

    def test_get_nns_by_item(self):
        f = 2
        i = AnnoyIndex(f, 'manhattan')
        i.add_item(0, [2, 2])
        i.add_item(1, [3, 2])
        i.add_item(2, [3, 3])
        i.build(10)

        self.assertEqual(i.get_nns_by_item(0, 3), [0, 1, 2])
        self.assertEqual(i.get_nns_by_item(2, 3), [2, 1, 0])

    def test_dist(self):
        f = 2
        i = AnnoyIndex(f, 'manhattan')
        i.add_item(0, [0, 1])
        i.add_item(1, [1, 1])
        i.add_item(2, [0, 0])

        self.assertAlmostEqual(i.get_distance(0, 1), 1.0)
        self.assertAlmostEqual(i.get_distance(1, 2), 2.0)

    def test_large_index(self):
        # Generate pairs of random points where the pair is super close
        f = 10
        i = AnnoyIndex(f, 'manhattan')
        for j in range(0, 10000, 2):
            p = [random.gauss(0, 1) for z in range(f)]
            x = [1 + pi + random.gauss(0, 1e-2) for pi in p]
            y = [1 + pi + random.gauss(0, 1e-2) for pi in p]
            i.add_item(j, x)
            i.add_item(j+1, y)

        i.build(10)
        for j in range(0, 10000, 2):
            self.assertEqual(i.get_nns_by_item(j, 2), [j, j+1])
            self.assertEqual(i.get_nns_by_item(j+1, 2), [j+1, j])

    def precision(self, n, n_trees=10, n_points=10000, n_rounds=10):
        found = 0
        for r in range(n_rounds):
            # create random points at distance x
            f = 10
            i = AnnoyIndex(f, 'manhattan')
            for j in range(n_points):
                p = [random.gauss(0, 1) for z in range(f)]
                norm = sum([pi ** 2 for pi in p]) ** 0.5
                x = [pi / norm + j for pi in p]
                i.add_item(j, x)

            i.build(n_trees)

            nns = i.get_nns_by_vector([0] * f, n)
            self.assertEqual(nns, sorted(nns))  # should be in order
            # The number of gaps should be equal to the last item minus n-1
            found += len([x for x in nns if x < n])

        return 1.0 * found / (n * n_rounds)

    def test_precision_1(self):
        self.assertTrue(self.precision(1) >= 0.98)

    def test_precision_10(self):
        self.assertTrue(self.precision(10) >= 0.98)

    def test_precision_100(self):
        self.assertTrue(self.precision(100) >= 0.98)

    def test_precision_1000(self):
        self.assertTrue(self.precision(1000) >= 0.98)

    def test_get_nns_with_distances(self):
        f = 3
        i = AnnoyIndex(f, 'manhattan')
        i.add_item(0, [0, 0, 2])
        i.add_item(1, [0, 1, 1])
        i.add_item(2, [1, 0, 0])
        i.build(10)

        l, d = i.get_nns_by_item(0, 3, -1, True)
        self.assertEqual(l, [0, 1, 2])
        self.assertAlmostEqual(d[0], 0.0)
        self.assertAlmostEqual(d[1], 2.0)
        self.assertAlmostEqual(d[2], 3.0)

        l, d = i.get_nns_by_vector([2, 2, 1], 3, -1, True)
        self.assertEqual(l, [1, 2, 0])
        self.assertAlmostEqual(d[0], 3.0)
        self.assertAlmostEqual(d[1], 4.0)
        self.assertAlmostEqual(d[2], 5.0)

    def test_include_dists(self):
        f = 40
        i = AnnoyIndex(f, 'manhattan')
        v = numpy.random.normal(size=f)
        i.add_item(0, v)
        i.add_item(1, -v)
        i.build(10)

        indices, dists = i.get_nns_by_item(0, 2, 10, True)
        self.assertEqual(indices, [0, 1])
        self.assertAlmostEqual(dists[0], 0.0)

    def test_distance_consistency(self):
        n, f = 1000, 3
        i = AnnoyIndex(f, 'manhattan')
        for j in range(n):
            i.add_item(j, numpy.random.normal(size=f))
        i.build(10)
        for a in random.sample(range(n), 100):
            indices, dists = i.get_nns_by_item(a, 100, include_distances=True)
            for b, dist in zip(indices, dists):
                self.assertAlmostEqual(dist, i.get_distance(a, b))
                u = numpy.array(i.get_item_vector(a))
                v = numpy.array(i.get_item_vector(b))
                self.assertAlmostEqual(dist, numpy.sum(numpy.fabs(u - v)))
                self.assertAlmostEqual(dist, sum([abs(float(x)-float(y)) for x, y in zip(u, v)]))
