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
    def test_random_holes(self):
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

    def _test_holes_base(self, n, f=100, base_i=100000):
        annoy = AnnoyIndex(f)
        for i in range(n):
            annoy.add_item(base_i + i, numpy.random.normal(size=(f,)))
        annoy.build(100)
        res = annoy.get_nns_by_item(base_i, n)
        self.assertEquals(set(res), set([base_i + i for i in range(n)]))

    def test_root_one_child(self):
        # See https://github.com/spotify/annoy/issues/223
        self._test_holes_base(1)

    def test_root_two_children(self):
        self._test_holes_base(2)

    def test_root_some_children(self):
        # See https://github.com/spotify/annoy/issues/295
        self._test_holes_base(10)

    def test_root_many_children(self):
        self._test_holes_base(1000)
