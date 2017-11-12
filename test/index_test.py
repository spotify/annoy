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

import random
from common import TestCase
from annoy import AnnoyIndex

class IndexTest(TestCase):
    def test_not_found_tree(self):
        i = AnnoyIndex(10)
        self.assertRaises(IOError, i.load, 'nonexists.tree')

    def test_binary_compatibility(self):
        i = AnnoyIndex(10)
        i.load('test/test.tree')

        # This might change in the future if we change the search algorithm, but in that case let's update the test
        self.assertEqual(i.get_nns_by_item(0, 10), [0, 85, 42, 11, 54, 38, 53, 66, 19, 31])

    def test_load_unload(self):
        # Issue #108
        i = AnnoyIndex(10)
        for x in range(100000):
            i.load('test/test.tree')
            i.unload()

    def test_construct_load_destruct(self):
        for x in range(100000):
            i = AnnoyIndex(10)
            i.load('test/test.tree')

    def test_construct_destruct(self):
        for x in range(100000):
            i = AnnoyIndex(10)
            i.add_item(1000, [random.gauss(0, 1) for z in range(10)])

    def test_save_twice(self):
        # Issue #100
        t = AnnoyIndex(10)
        t.save("t.ann")
        t.save("t.ann")

    def test_load_save(self):
        # Issue #61
        i = AnnoyIndex(10)
        i.load('test/test.tree')
        u = i.get_item_vector(99)
        i.save('x.tree')
        v = i.get_item_vector(99)
        self.assertEqual(u, v)
        j = AnnoyIndex(10)
        j.load('test/test.tree')
        w = i.get_item_vector(99)
        self.assertEqual(u, w)

    def test_save_without_build(self):
        # Issue #61
        i = AnnoyIndex(10)
        i.add_item(1000, [random.gauss(0, 1) for z in range(10)])
        i.save('x.tree')
        j = AnnoyIndex(10)
        j.load('x.tree')
        j.build(10)
        
    def test_unbuild_with_loaded_tree(self):
        i = AnnoyIndex(10)
        i.load('test/test.tree')
        i.unbuild()

    def test_seed(self):
        i = AnnoyIndex(10)
        i.load('test/test.tree')
        i.set_seed(42)

    def test_unknown_distance(self):
        self.assertRaises(Exception, AnnoyIndex, 10, 'banana')

    def test_metric_kwarg(self):
        # Issue 211
        i = AnnoyIndex(2, metric='euclidean')
        i.add_item(0, [1, 0])
        i.add_item(1, [9, 0])
        self.assertAlmostEqual(i.get_distance(0, 1), 8)
        self.assertEquals(i.f, 2)

    def test_metric_f_kwargs(self):
        i = AnnoyIndex(f=3, metric='euclidean')
