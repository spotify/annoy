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

# TeamCity fails to run this test because it can't import the C++ module.
# I think it's because the C++ part gets built in another directory.

import unittest
import random
from annoy import AnnoyIndex


class AngularIndexTest(unittest.TestCase):
    def test_get_nns_by_vector(self):
        f = 3
        i = AnnoyIndex(f)
        i.add_item(0, [1,0,0])
        i.add_item(1, [0,1,0])
        i.add_item(2, [0,0,1])
        i.build(10)

        self.assertEqual(i.get_nns_by_vector([3,2,1], 3), [0,1,2])

    def test_get_nns_by_item(self):
        f = 3
        i = AnnoyIndex(f)
        i.add_item(0, [2,1,0])
        i.add_item(1, [1,2,0])
        i.add_item(2, [0,0,1])
        i.build(10)

        self.assertEqual(i.get_nns_by_item(0, 3), [0,1,2])
        self.assertEqual(i.get_nns_by_item(1, 3), [1,0,2])

    def test_dist(self):
        f = 2
        i = AnnoyIndex(f)
        i.add_item(0, [0, 1])
        i.add_item(1, [1, 1])

        self.assertAlmostEqual(i.get_distance(0, 1), 2 * (1.0 - 2 ** -0.5))

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

        dist = (1 - 2 ** -0.5) ** 2 + (2 ** -0.5) ** 2

        self.assertAlmostEqual(i.get_distance(0, 1), dist)

    def test_dist_degen(self):
        f = 2
        i = AnnoyIndex(f)
        i.add_item(0, [1, 0])
        i.add_item(1, [0, 0])

        self.assertAlmostEqual(i.get_distance(0, 1), 2.0)

    def test_large_index(self):
        # Generate pairs of random points where the pair is super close
        f = 10
        i = AnnoyIndex(f)
        for j in xrange(0, 10000, 2):
            p = [random.gauss(0, 1) for z in xrange(f)]
            f1 = random.random() + 1
            f2 = random.random() + 1
            x = [f1 * pi + random.gauss(0, 1e-2) for pi in p]
            y = [f2 * pi + random.gauss(0, 1e-2) for pi in p]
            i.add_item(j, x)
            i.add_item(j+1, y)

        i.build(10)
        for j in xrange(0, 10000, 2):
            self.assertEqual(i.get_nns_by_item(j, 2), [j, j+1])
            self.assertEqual(i.get_nns_by_item(j+1, 2), [j+1, j])


class EuclideanIndexTest(unittest.TestCase):
    def test_get_nns_by_vector(self):
        f = 2
        i = AnnoyIndex(f, 'euclidean')
        i.add_item(0, [2,2])
        i.add_item(1, [3,2])
        i.build(10)

        self.assertEqual(i.get_nns_by_vector([3,3], 2), [1, 0])

    def test_dist(self):
        f = 2
        i = AnnoyIndex(f, 'euclidean')
        i.add_item(0, [0, 1])
        i.add_item(1, [1, 1])

        self.assertAlmostEqual(i.get_distance(0, 1), 1.0)

    def test_large_index(self):
        # Generate pairs of random points where the pair is super close
        f = 10
        q = [random.gauss(0, 10) for z in xrange(f)]
        i = AnnoyIndex(f, 'euclidean')
        for j in xrange(0, 10000, 2):
            p = [random.gauss(0, 1) for z in xrange(f)]
            x = [1 + pi + random.gauss(0, 1e-2) for pi in p] # todo: should be q[i]
            y = [1 + pi + random.gauss(0, 1e-2) for pi in p]
            i.add_item(j, x)
            i.add_item(j+1, y)

        i.build(10)
        for j in xrange(0, 10000, 2):
            self.assertEqual(i.get_nns_by_item(j, 2), [j, j+1])
            self.assertEqual(i.get_nns_by_item(j+1, 2), [j+1, j])

    def precision(self, n, n_trees=10, n_points=10000):
        # create random points at distance x
        f = 10
        i = AnnoyIndex(f, 'euclidean')
        for j in xrange(n_points):
            p = [random.gauss(0, 1) for z in xrange(f)]
            norm = sum([pi ** 2 for pi in p]) ** 0.5
            x = [pi / norm * j for pi in p]
            i.add_item(j, x)

        i.build(n_trees)

        nns = i.get_nns_by_vector([0] * f, n)
        self.assertEqual(nns, sorted(nns))  # should be in order
        # The number of gaps should be equal to the last item minus n-1
        found = len([x for x in nns if x < n])
        return 1.0 * found / n

    def test_precision_1(self):
        self.assertEqual(self.precision(1), 1.0)

    def test_precision_10(self):
        self.assertEqual(self.precision(10), 1.0)

    def test_precision_100(self):
        self.assertGreaterEqual(self.precision(100), 0.99)

    def test_precision_1000(self):
        self.assertGreaterEqual(self.precision(1000), 0.99)

    def test_not_found_tree(self):
        i = AnnoyIndex(10)
        with self.assertRaises(IOError):
            i.load("nonexists.tree")
