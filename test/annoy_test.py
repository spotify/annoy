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
import numpy
from annoy import AnnoyIndex

try:
    xrange
except NameError:
    # Python 3 compat
    xrange = range


class TestCase(unittest.TestCase):
    def assertAlmostEquals(self, x, y):
        # Annoy uses float precision, so we override the default precision
        super(TestCase, self).assertAlmostEquals(x, y, 4)


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

    def precision(self, n, n_trees=10, n_points=10000, n_rounds=10):
        found = 0
        for r in xrange(n_rounds):
            # create random points at distance x from (1000, 0, 0, ...)
            f = 10
            i = AnnoyIndex(f, 'euclidean')
            for j in xrange(n_points):
                p = [random.gauss(0, 1) for z in xrange(f - 1)]
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


class EuclideanIndexTest(TestCase):
    def test_get_nns_by_vector(self):
        f = 2
        i = AnnoyIndex(f, 'euclidean')
        i.add_item(0, [2, 2])
        i.add_item(1, [3, 2])
        i.add_item(2, [3, 3])
        i.build(10)

        self.assertEqual(i.get_nns_by_vector([4, 4], 3), [2, 1, 0])
        self.assertEqual(i.get_nns_by_vector([1, 1], 3), [0, 1, 2])
        self.assertEqual(i.get_nns_by_vector([4, 2], 3), [1, 2, 0])

    def test_get_nns_by_item(self):
        f = 2
        i = AnnoyIndex(f, 'euclidean')
        i.add_item(0, [2, 2])
        i.add_item(1, [3, 2])
        i.add_item(2, [3, 3])
        i.build(10)

        self.assertEqual(i.get_nns_by_item(0, 3), [0, 1, 2])
        self.assertEqual(i.get_nns_by_item(2, 3), [2, 1, 0])

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

    def precision(self, n, n_trees=10, n_points=10000, n_rounds=10):
        found = 0
        for r in xrange(n_rounds):
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
        i = AnnoyIndex(f, 'euclidean')
        i.add_item(0, [0, 0, 2])
        i.add_item(1, [0, 1, 1])
        i.add_item(2, [1, 0, 0])
        i.build(10)

        l, d = i.get_nns_by_item(0, 3, -1, True)
        self.assertEquals(l, [0, 1, 2])
        self.assertAlmostEquals(d[0]**2, 0.0)
        self.assertAlmostEquals(d[1]**2, 2.0)
        self.assertAlmostEquals(d[2]**2, 5.0)

        l, d = i.get_nns_by_vector([2, 2, 2], 3, -1, True)
        self.assertEquals(l, [1, 0, 2])
        self.assertAlmostEquals(d[0]**2, 6.0)
        self.assertAlmostEquals(d[1]**2, 8.0)
        self.assertAlmostEquals(d[2]**2, 9.0)


class IndexTest(TestCase):
    def test_not_found_tree(self):
        i = AnnoyIndex(10)
        self.assertRaises(IOError, i.load, 'nonexists.tree')

    def test_binary_compatibility(self):
        i = AnnoyIndex(10)
        i.load('test/test.tree')

        # This might change in the future if we change the search algorithm, but in that case let's update the test
        self.assertEquals(i.get_nns_by_item(0, 10), [0, 85, 42, 11, 54, 38, 53, 66, 19, 31])


class TypesTest(TestCase):
    def test_numpy(self, n_points=1000, n_trees=10):
        f = 10
        i = AnnoyIndex(f, 'euclidean')
        for j in xrange(n_points):
            a = numpy.random.normal(size=f)
            a = a.astype(random.choice([numpy.float64, numpy.float32, numpy.uint8, numpy.int16]))
            i.add_item(j, a)

        i.build(n_trees)

    def test_tuple(self, n_points=1000, n_trees=10):
        f = 10
        i = AnnoyIndex(f, 'euclidean')
        for j in xrange(n_points):
            i.add_item(j, (random.gauss(0, 1) for x in xrange(f)))

        i.build(n_trees)

    def test_wrong_length(self, n_points=1000, n_trees=10):
        f = 10
        i = AnnoyIndex(f, 'euclidean')
        i.add_item(0, [random.gauss(0, 1) for x in xrange(f)])
        self.assertRaises(IndexError, i.add_item, 1, [random.gauss(0, 1) for x in xrange(f+1000)])
        self.assertRaises(IndexError, i.add_item, 2, [])

        i.build(n_trees)
