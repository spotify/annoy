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
from nose.plugins.skip import SkipTest


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

    @SkipTest  # will fix later
    def test_zero_vectors(self):
        # Mentioned on the annoy-user list
        bitstrings = [
            '0000000000011000001110000011111000101110111110000100000100000000',
            '0000000000011000001110000011111000101110111110000100000100000001',
            '0000000000011000001110000011111000101110111110000100000100000010',
            '0010010100011001001000010001100101011110000000110000011110001100',
            '1001011010000110100101101001111010001110100001101000111000001110',
            '0111100101111001011110010010001100010111000111100001101100011111',
            '0011000010011101000011010010111000101110100101111000011101001011',
            '0011000010011100000011010010111000101110100101111000011101001011',
            '1001100000111010001010000010110000111100100101001001010000000111',
            '0000000000111101010100010001000101101001000000011000001101000000',
            '1000101001010001011100010111001100110011001100110011001111001100',
            '1110011001001111100110010001100100001011000011010010111100100111',
        ]
        vectors = [[int(bit) for bit in bitstring] for bitstring in bitstrings]

        f = 64
        idx = AnnoyIndex(f, 'hamming')
        for i, v in enumerate(vectors):
            idx.add_item(i, v)

        idx.build(10)
        idx.save('idx.ann')
        idx = AnnoyIndex(f, 'hamming')
        idx.load('idx.ann')
        js, ds = idx.get_nns_by_item(0, 5, include_distances=True)
        self.assertEquals(js[0], 0)
        self.assertEquals(ds[:4], [0, 1, 1, 22])
