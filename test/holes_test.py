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


class HolesTest(TestCase):
    # See https://github.com/spotify/annoy/issues/223
    def test_holes(self):
        f = 10
        index = AnnoyIndex(f)
        index.add_item(1000, numpy.random.normal(size=(f,)))
        index.build(10)
        js = index.get_nns_by_vector(numpy.random.normal(size=(f,)), 100)
        self.assertEquals(js, [1000])

    def test_holes_more(self):
        f = 10
        index = AnnoyIndex(f)
        valid_indices = random.sample(range(2000), 1000) # leave holes
        for i in valid_indices:
            v = numpy.random.normal(size=(f,))
            index.add_item(i, v)
        index.build(10)
        for i in valid_indices:
            js = index.get_nns_by_item(i, 10000)
            for j in js:
                self.assertTrue(j in valid_indices)
        for i in range(1000):
            v = numpy.random.normal(size=(f,))
            js = index.get_nns_by_vector(v, 10000)
            for j in js:
                self.assertTrue(j in valid_indices)
