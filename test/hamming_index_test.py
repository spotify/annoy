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


class HammingIndexTest(TestCase):
    def test_basic_conversion(self):
        f = 100
        i = AnnoyIndex(f, 'hamming')
        u = numpy.random.binomial(1, 0.5, f)
        v = numpy.random.binomial(1, 0.5, f)
        i.add_item(0, u)
        i.add_item(1, v)
        u2 = i.get_item_vector(0)
        v2 = i.get_item_vector(1)
        self.assertAlmostEqual(numpy.dot(u - u2, u - u2), 0.0)
        self.assertAlmostEqual(numpy.dot(v - v2, v - v2), 0.0)
        self.assertAlmostEqual(i.get_distance(0, 0), 0.0)
        self.assertAlmostEqual(i.get_distance(1, 1), 0.0)
        self.assertAlmostEqual(i.get_distance(0, 1), numpy.dot(u - v, u - v))
        self.assertAlmostEqual(i.get_distance(1, 0), numpy.dot(u - v, u - v))

    def test_basic_nns(self):
        f = 100
        i = AnnoyIndex(f, 'hamming')
        u = numpy.random.binomial(1, 0.5, f)
        v = numpy.random.binomial(1, 0.5, f)
        i.add_item(0, u)
        i.add_item(1, v)
        i.build(10)
        self.assertEquals(i.get_nns_by_item(0, 99), [0, 1])
        self.assertEquals(i.get_nns_by_item(1, 99), [1, 0])
        rs, ds = i.get_nns_by_item(0, 99, include_distances=True)
        self.assertEquals(rs, [0, 1])
        self.assertAlmostEqual(ds[0], 0)
        self.assertAlmostEqual(ds[1], numpy.dot(u-v, u-v))

    def test_save_load(self):
        f = 100
        i = AnnoyIndex(f, 'hamming')
        u = numpy.random.binomial(1, 0.5, f)
        v = numpy.random.binomial(1, 0.5, f)
        i.add_item(0, u)
        i.add_item(1, v)
        i.build(10)
        i.save('blah.ann')
        j = AnnoyIndex(f, 'hamming')
        j.load('blah.ann')
        rs, ds = j.get_nns_by_item(0, 99, include_distances=True)
        self.assertEquals(rs, [0, 1])
        self.assertAlmostEqual(ds[0], 0)
        self.assertAlmostEqual(ds[1], numpy.dot(u-v, u-v))

    def test_many_vectors(self):
        f = 10
        i = AnnoyIndex(f, 'hamming')
        for x in range(100000):
            i.add_item(x, numpy.random.binomial(1, 0.5, f))
        i.build(10)

        rs, ds = i.get_nns_by_vector([0]*f, 10000, include_distances=True)
        self.assertGreaterEqual(min(ds), 0)
        self.assertLessEqual(max(ds), f)

        dists = []
        for x in range(1000):
            rs, ds = i.get_nns_by_vector(numpy.random.binomial(1, 0.5, f), 1, search_k=1000, include_distances=True)
            dists.append(ds[0])
        avg_dist = 1.0 * sum(dists) / len(dists)
        self.assertLessEqual(avg_dist, 0.42)
