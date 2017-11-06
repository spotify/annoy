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
