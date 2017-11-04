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
from common import TestCase
from annoy import AnnoyIndex


class SeedTest(TestCase):
    def test_seeding(self):
        f = 10
        X = numpy.random.rand(1000, f)
        Y = numpy.random.rand(50, f)

        indexes = []
        for i in range(2):
            index = AnnoyIndex(f)
            index.set_seed(42)
            for j in range(X.shape[0]):
                index.add_item(j, X[j])

            index.build(10)
            indexes.append(index)

        for k in range(Y.shape[0]):
            self.assertEquals(indexes[0].get_nns_by_vector(Y[k], 100),
                              indexes[1].get_nns_by_vector(Y[k], 100))

