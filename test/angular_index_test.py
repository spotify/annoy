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


class AngularIndexTest(TestCase):
    def test_get_nns_by_vector(self):
        f = 3
        i = AnnoyIndex(f)
        i.add_item(0, [0, 0, 1])
        i.add_item(1, [0, 1, 0])
        i.add_item(2, [1, 0, 0])
        i.build(10)

        self.assertEqual(i.get_nns_by_vector([3, 2, 1], 3), [2, 1, 0])
        self.assertEqual(i.get_nns_by_vector([1, 2, 3], 3), [0, 1, 2])
        self.assertEqual(i.get_nns_by_vector([2, 0, 1], 3), [2, 0, 1])

    def test_get_nns_by_item(self):
        f = 3
        i = AnnoyIndex(f)
        i.add_item(0, [2, 1, 0])
        i.add_item(1, [1, 2, 0])
        i.add_item(2, [0, 0, 1])
        i.build(10)

        self.assertEqual(i.get_nns_by_item(0, 3), [0, 1, 2])
        self.assertEqual(i.get_nns_by_item(1, 3), [1, 0, 2])
        self.assertTrue(i.get_nns_by_item(2, 3) in [[2, 0, 1], [2, 1, 0]]) # could be either

    def test_dist(self):
        f = 2
        i = AnnoyIndex(f)
        i.add_item(0, [0, 1])
        i.add_item(1, [1, 1])

        self.assertAlmostEqual(i.get_distance(0, 1), (2 * (1.0 - 2 ** -0.5))**0.5)

    def test_dist_2(self):
        f = 2
        i = AnnoyIndex(f)
        i.add_item(0, [1000, 0])
        i.add_item(1, [10, 0])

        self.assertAlmostEqual(i.get_distance(0, 1), 0)

    def test_dist_3(self):
        f = 2
        i = AnnoyIndex(f)
        i.add_item(0, [97, 0])
        i.add_item(1, [42, 42])

        dist = ((1 - 2 ** -0.5) ** 2 + (2 ** -0.5) ** 2)**0.5

        self.assertAlmostEqual(i.get_distance(0, 1), dist)

    def test_dist_degen(self):
        f = 2
        i = AnnoyIndex(f)
        i.add_item(0, [1, 0])
        i.add_item(1, [0, 0])

        self.assertAlmostEqual(i.get_distance(0, 1), 2.0**0.5)

    def test_large_index(self):
        # Generate pairs of random points where the pair is super close
        f = 10
        i = AnnoyIndex(f)
        for j in range(0, 10000, 2):
            p = [random.gauss(0, 1) for z in range(f)]
            f1 = random.random() + 1
            f2 = random.random() + 1
            x = [f1 * pi + random.gauss(0, 1e-2) for pi in p]
            y = [f2 * pi + random.gauss(0, 1e-2) for pi in p]
            i.add_item(j, x)
            i.add_item(j+1, y)

        i.build(10)
        for j in range(0, 10000, 2):
            self.assertEqual(i.get_nns_by_item(j, 2), [j, j+1])
            self.assertEqual(i.get_nns_by_item(j+1, 2), [j+1, j])

    def precision(self, n, n_trees=10, n_points=10000, n_rounds=10):
        found = 0
        for r in range(n_rounds):
            # create random points at distance x from (1000, 0, 0, ...)
            f = 10
            i = AnnoyIndex(f, 'euclidean')
            for j in range(n_points):
                p = [random.gauss(0, 1) for z in range(f - 1)]
                norm = sum([pi ** 2 for pi in p]) ** 0.5
                x = [1000] + [pi / norm * j for pi in p]
                i.add_item(j, x)

            i.build(n_trees)

            nns = i.get_nns_by_vector([1000] + [0] * (f-1), n)
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

    def test_load_save_get_item_vector(self):
        f = 3
        i = AnnoyIndex(f)
        i.add_item(0, [1.1, 2.2, 3.3])
        i.add_item(1, [4.4, 5.5, 6.6])
        i.add_item(2, [7.7, 8.8, 9.9])
 
        numpy.testing.assert_array_almost_equal(i.get_item_vector(0), [1.1, 2.2, 3.3])
        self.assertTrue(i.build(10))
        self.assertTrue(i.save('blah.ann'))
        numpy.testing.assert_array_almost_equal(i.get_item_vector(1), [4.4, 5.5, 6.6])
        j = AnnoyIndex(f)
        self.assertTrue(j.load('blah.ann'))
        numpy.testing.assert_array_almost_equal(j.get_item_vector(2), [7.7, 8.8, 9.9])

    def test_get_nns_search_k(self):
        f = 3
        i = AnnoyIndex(f)
        i.add_item(0, [0, 0, 1])
        i.add_item(1, [0, 1, 0])
        i.add_item(2, [1, 0, 0])
        i.build(10)

        self.assertEqual(i.get_nns_by_item(0, 3, 10), [0, 1, 2])
        self.assertEqual(i.get_nns_by_vector([3, 2, 1], 3, 10), [2, 1, 0])

    def test_include_dists(self):
        # Double checking issue 112
        f = 40
        i = AnnoyIndex(f)
        v = numpy.random.normal(size=f)
        i.add_item(0, v)
        i.add_item(1, -v)
        i.build(10)

        indices, dists = i.get_nns_by_item(0, 2, 10, True)
        self.assertEqual(indices, [0, 1])
        self.assertAlmostEqual(dists[0], 0.0)
        self.assertAlmostEqual(dists[1], 2.0)

    def test_include_dists_check_ranges(self):
        f = 3
        i = AnnoyIndex(f)
        for j in range(100000):
            i.add_item(j, numpy.random.normal(size=f))
        i.build(10)
        indices, dists = i.get_nns_by_item(0, 100000, include_distances=True)
        self.assertTrue(max(dists) < 2.0)
        self.assertAlmostEqual(min(dists), 0.0)

    def test_distance_consistency(self):
        n, f = 1000, 3
        i = AnnoyIndex(f)
        for j in range(n):
            i.add_item(j, numpy.random.normal(size=f))
        i.build(10)
        for a in random.sample(range(n), 100):
            indices, dists = i.get_nns_by_item(a, 100, include_distances=True)
            for b, dist in zip(indices, dists):
                self.assertAlmostEqual(dist, i.get_distance(a, b))
                u = i.get_item_vector(a)
                v = i.get_item_vector(b)
                u_norm = numpy.array(u) * numpy.dot(u, u)**-0.5
                v_norm = numpy.array(v) * numpy.dot(v, v)**-0.5
                # cos = numpy.clip(1 - cosine(u, v), -1, 1) # scipy returns 1 - cos
                self.assertAlmostEqual(dist, numpy.dot(u_norm - v_norm, u_norm - v_norm) ** 0.5)
                # self.assertAlmostEqual(dist, (2*(1 - cos))**0.5)
                self.assertAlmostEqual(dist, sum([(x-y)**2 for x, y in zip(u_norm, v_norm)])**0.5)

    def test_only_one_item(self):
        # reported to annoy-user by Kireet Reddy
        idx = AnnoyIndex(100)
        idx.add_item(0, numpy.random.randn(100))
        idx.build(n_trees=10)
        idx.save('foo.idx')
        idx = AnnoyIndex(100)
        idx.load('foo.idx')
        self.assertEquals(idx.get_n_items(), 1)
        self.assertEquals(idx.get_nns_by_vector(vector=numpy.random.randn(100), n=50, include_distances=False), [0])

    def test_no_items(self):
        idx = AnnoyIndex(100)
        idx.build(n_trees=10)
        idx.save('foo.idx')
        idx = AnnoyIndex(100)
        idx.load('foo.idx')
        self.assertEquals(idx.get_n_items(), 0)
        self.assertEquals(idx.get_nns_by_vector(vector=numpy.random.randn(100), n=50, include_distances=False), [])

    def test_single_vector(self):
        # https://github.com/spotify/annoy/issues/194
        a = AnnoyIndex(3)
        a.add_item(0, [1, 0, 0])
        a.build(10)
        a.save('1.ann')
        self.assertEquals(a.get_nns_by_vector([1, 0, 0], 3, include_distances=True), ([0], [0.0]))
    
