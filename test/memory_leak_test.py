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


class MemoryLeakTest(TestCase):
    def test_get_item_vector(self):
        f = 10
        i = AnnoyIndex(f, 'euclidean')
        i.add_item(0, [random.gauss(0, 1) for x in range(f)])
        for j in range(100):
            print(j, '...')
            for k in range(1000 * 1000):
                i.get_item_vector(0)

    def test_get_lots_of_nns(self):
        f = 10
        i = AnnoyIndex(f, 'euclidean')
        i.add_item(0, [random.gauss(0, 1) for x in range(f)])
        i.build(10)
        for j in range(100):
            self.assertEqual(i.get_nns_by_item(0, 999999999), [0])
            
    def test_build_unbuid(self):
        f = 10
        i = AnnoyIndex(f, 'euclidean')
        for j in range(1000):
            i.add_item(j, [random.gauss(0, 1) for x in range(f)])
        i.build(10)
        
        for j in range(100):
            i.unbuild()
            i.build(10)
            
        self.assertEqual(i.get_n_items(), 1000)

