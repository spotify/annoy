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

import os
import sys
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
        i.save('i.tree')
        v = i.get_item_vector(99)
        self.assertEqual(u, v)
        j = AnnoyIndex(10)
        j.load('test/test.tree')
        w = i.get_item_vector(99)
        self.assertEqual(u, w)
        # Ensure specifying if prefault is allowed does not impact result
        j.save('j.tree', True)
        k = AnnoyIndex(10)
        k.load('j.tree', True)
        x = k.get_item_vector(99)
        self.assertEqual(u, x)
        k.save('k.tree', False)
        l = AnnoyIndex(10)
        l.load('k.tree', False)
        y = l.get_item_vector(99)
        self.assertEqual(u, y)

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
        self.assertEqual(i.f, 2)

    def test_metric_f_kwargs(self):
        i = AnnoyIndex(f=3, metric='euclidean')

    def test_item_vector_after_save(self):
        # Issue #279
        a = AnnoyIndex(3)
        a.verbose(True)
        a.add_item(1, [1, 0, 0])
        a.add_item(2, [0, 1, 0])
        a.add_item(3, [0, 0, 1])
        a.build(-1)
        self.assertEqual(a.get_n_items(), 4)
        self.assertEqual(a.get_item_vector(3), [0, 0, 1])
        self.assertEqual(set(a.get_nns_by_item(1, 999)), set([1, 2, 3]))
        a.save('something.annoy')
        self.assertEqual(a.get_n_items(), 4)
        self.assertEqual(a.get_item_vector(3), [0, 0, 1])
        self.assertEqual(set(a.get_nns_by_item(1, 999)), set([1, 2, 3]))

    def test_prefault(self):
        i = AnnoyIndex(10)
        i.load('test/test.tree', prefault=True)
        self.assertEqual(i.get_nns_by_item(0, 10), [0, 85, 42, 11, 54, 38, 53, 66, 19, 31])

    def test_fail_save(self):
        t = AnnoyIndex(40)
        with self.assertRaises(IOError):
            t.save('')

    def test_overwrite_index(self):
        # Issue #335
        f = 40

        # Build the initial index
        t = AnnoyIndex(f)
        for i in range(1000):
            v = [random.gauss(0, 1) for z in range(f)]
            t.add_item(i, v)
        t.build(10)
        t.save('test.ann')

        # Load index file
        t2 = AnnoyIndex(f)
        t2.load('test.ann')

        # Overwrite index file
        t3 = AnnoyIndex(f)
        for i in range(500):
            v = [random.gauss(0, 1) for z in range(f)]
            t3.add_item(i, v)
        t3.build(10)
        if os.name == 'nt':
            # Can't overwrite on Windows
            with self.assertRaises(IOError):
                t3.save('test.ann')
        else:
            t3.save('test.ann')
            # Get nearest neighbors
            v = [random.gauss(0, 1) for z in range(f)]
            nns = t2.get_nns_by_vector(v, 1000)  # Should not crash

    def test_get_n_trees(self):
        i = AnnoyIndex(10)
        i.load('test/test.tree')
        self.assertEqual(i.get_n_trees(), 10)

    def test_write_failed(self):
        f = 40

        # Build the initial index
        t = AnnoyIndex(f)
        for i in range(1000):
            v = [random.gauss(0, 1) for z in range(f)]
            t.add_item(i, v)
        t.build(10)

        if sys.platform == "linux" or sys.platform == "linux2":
            # linux
            try:
                t.save("/dev/full") 
                self.fail("didn't get expected exception")
            except Exception as e:
                self.assertTrue(str(e).find("No space left on device") > 0)
        elif sys.platform == "darwin":
            volume = "FULLDISK"
            device = os.popen('hdiutil attach -nomount ram://64').read()
            os.popen('diskutil erasevolume MS-DOS %s %s' % (volume, device))
            os.popen('touch "/Volumes/%s/full"' % volume)
            try:
                t.save('/Volumes/%s/annoy.tree' % volume)
                self.fail("didn't get expected exception")
            except Exception as e:
                self.assertTrue(str(e).find("No space left on device") > 0)
            finally:
                os.popen("hdiutil detach %s" % device)
