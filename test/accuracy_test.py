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

from __future__ import print_function

import os

import h5py

from annoy import AnnoyIndex

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve  # Python 3


def _get_index(dataset, custom_distance=None, custom_dim=None):
    url = 'http://ann-benchmarks.com/%s.hdf5' % dataset
    vectors_fn = os.path.join("test", dataset + ".hdf5")
    index_fn = os.path.join("test", dataset + ".annoy")

    if not os.path.exists(vectors_fn):
        print("downloading", url, "->", vectors_fn)
        urlretrieve(url, vectors_fn)

    dataset_f = h5py.File(vectors_fn, "r")
    distance = dataset_f.attrs["distance"]
    if custom_distance is not None:
         distance = custom_distance
    f = dataset_f["train"].shape[1]
    if custom_dim:
         f = custom_dim
    if custom_distance:
        dataset = dataset.rsplit('-', 2)[0] + "-%d-%s" % (f, custom_distance)
        index_fn = os.path.join('test', dataset + '.annoy')


    annoy = AnnoyIndex(f, distance)

    if not os.path.exists(index_fn):
        print("adding items", distance, f)
        for i, v in enumerate(dataset_f["train"]):
            if len(v) > f:
                v = v[:f]
            annoy.add_item(i, v)

        print("building index")
        annoy.build(10)
        annoy.save(index_fn)
    else:
        annoy.load(index_fn)
    return annoy, dataset_f, dataset


def _test_index(dataset, exp_accuracy, custom_metric=None, custom_dim=None):
    annoy, dataset_f, dataset = _get_index(dataset, custom_metric, custom_dim)

    n, k = 0, 0

    for i, v in enumerate(dataset_f["test"]):
        if custom_dim:
            v = v[:custom_dim]
        js_fast = annoy.get_nns_by_vector(v, 10, 10000)
        js_real = dataset_f["neighbors"][i][:10]
        assert len(js_fast) == 10
        assert len(js_real) == 10

        n += 10
        k += len(set(js_fast).intersection(js_real))

    accuracy = 100.0 * k / n
    print(
        "%50s accuracy: %5.2f%% (expected %5.2f%%)" % (dataset, accuracy, exp_accuracy)
    )


    assert accuracy > exp_accuracy - 1.0  # should be within 1%


def test_glove_25():
    _test_index("glove-25-angular", 69.00)


def test_nytimes_16():
    _test_index("nytimes-16-angular", 80.00)


def test_lastfm_dot():
    _test_index('lastfm-64-dot', 60.00, 'dot', 64)


def test_lastfm_angular():
    _test_index('lastfm-64-dot', 60.00, 'angular', 65)
