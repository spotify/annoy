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
import os
import h5py
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve # Python 3

class SeedTest(TestCase):
    def test_seeding(self):
        f = 10
        X = numpy.random.rand(1000, f)
        Y = numpy.random.rand(50, f)

        indexes = []
        for i in range(2):
            index = AnnoyIndex(f, 'angular')
            index.set_seed(42)
            for j in range(X.shape[0]):
                index.add_item(j, X[j])

            index.build(10)
            indexes.append(index)

        for k in range(Y.shape[0]):
            self.assertEquals(indexes[0].get_nns_by_vector(Y[k], 100),
                              indexes[1].get_nns_by_vector(Y[k], 100))

    def test_historical(self):
        dataset = "fashion-mnist-784-euclidean"
        url = 'http://vectors.erikbern.com/%s.hdf5' % dataset
        vectors_fn = os.path.join('test', dataset + '.hdf5')
        if not os.path.exists(vectors_fn):
            urlretrieve(url, vectors_fn)

        dataset_f = h5py.File(vectors_fn, 'r')
        f = dataset_f['train'].shape[1]

        # Round 1, checking that we recover historical results with the default seed.
        annoy = AnnoyIndex(f, 'euclidean')
        for i, v in enumerate(dataset_f['train']):
            annoy.add_item(i, v)

        annoy.build(10)
        self.assertEquals(annoy.get_nns_by_item(0, 10),
                          [0, 27655, 18247, 9936, 48748, 26244, 49961, 38909, 55767, 38152])
        self.assertEquals(annoy.get_nns_by_item(10, 10),
                          [10, 48474, 15493, 18055, 54960, 32003, 13842, 24831, 24497, 26585])
        self.assertEquals(annoy.get_nns_by_item(100, 10),
                          [100, 19840, 28600, 4270, 49340, 38437, 26777, 57981, 25662, 46624])
        self.assertEquals(annoy.get_nns_by_item(1000, 10),
                          [1000, 55577, 20213, 379, 17810, 12464, 35858, 53879, 56078, 25054])
        self.assertEquals(annoy.get_nns_by_item(10000, 10),
                          [10000, 46719, 50444, 15672, 36818, 30426, 4374, 58080, 10938, 25192])

        # Round 2, checking that changing the seed actually does have an effect.
        annoy2 = AnnoyIndex(f, 'euclidean')
        for i, v in enumerate(dataset_f['train']):
            annoy2.add_item(i, v)

        annoy2.set_seed(0)
        annoy2.build(10)
        self.assertEquals(annoy2.get_nns_by_item(0, 10),
                          [0, 25719, 27655, 55310, 18247, 9936, 48748, 49961, 35683, 47527])
        self.assertEquals(annoy2.get_nns_by_item(10, 10),
                          [10, 48474, 15493, 18055, 54960, 32003, 14603, 13842, 24831, 24497])
        self.assertEquals(annoy2.get_nns_by_item(100, 10),
                          [100, 19840, 28600, 4270, 49340, 38437, 26777, 57981, 52911, 25662])
        self.assertEquals(annoy2.get_nns_by_item(1000, 10),
                          [1000, 55577, 20213, 379, 17810, 35858, 53879, 56078, 25054, 20816])
        self.assertEquals(annoy2.get_nns_by_item(10000, 10),
                          [10000, 50444, 15672, 36818, 30426, 4374, 58080, 25192, 47437, 25348])

