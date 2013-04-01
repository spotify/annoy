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

        self.assertEquals(i.get_nns_by_vector([3,2,1], 3), [0,1,2])

    def test_get_nns_by_item(self):
        f = 3
        i = AnnoyIndex(f)
        i.add_item(0, [2,1,0])
        i.add_item(1, [1,2,0])
        i.add_item(2, [0,0,1])
        i.build(10)

        self.assertEquals(i.get_nns_by_item(0, 3), [0,1,2])
        self.assertEquals(i.get_nns_by_item(1, 3), [1,0,2])

    def test_dist(self):
        f = 2
        i = AnnoyIndex(f)
        i.add_item(0, [0, 1])
        i.add_item(1, [1, 1])

        self.assertAlmostEqual(i.get_distance(0, 1), 1.0 - 2 ** -0.5)

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
            self.assertEquals(i.get_nns_by_item(j, 2), [j, j+1])
            self.assertEquals(i.get_nns_by_item(j+1, 2), [j+1, j])


class EuclideanIndexTest(unittest.TestCase):
    def test_get_nns_by_vector(self):
        f = 2
        i = AnnoyIndex(f, 'euclidean')
        i.add_item(0, [2,2])
        i.add_item(1, [3,2])
        i.build(10)

        self.assertEquals(i.get_nns_by_vector([3,3], 2), [1, 0])

    def test_dist(self):
        f = 2
        i = AnnoyIndex(f, 'euclidean')
        i.add_item(0, [0, 1])
        i.add_item(1, [1, 1])

        self.assertAlmostEqual(i.get_distance(0, 1), 1.0)

    def test_large_index(self):
        # Generate pairs of random points where the pair is super close
        f = 10
        i = AnnoyIndex(f, 'euclidean')
        for j in xrange(0, 10000, 2):
            p = [random.gauss(0, 1) for z in xrange(f)]
            x = [pi + random.gauss(0, 1e-2) for pi in p]
            y = [pi + random.gauss(0, 1e-2) for pi in p]
            i.add_item(j, x)
            i.add_item(j+1, y)
        
        i.build(10)
        for j in xrange(0, 10000, 2):
            self.assertEquals(i.get_nns_by_item(j, 2), [j, j+1])
            self.assertEquals(i.get_nns_by_item(j+1, 2), [j+1, j])
