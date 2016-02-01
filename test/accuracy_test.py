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

import unittest
import random
import os
from annoy import AnnoyIndex
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve # Python 3
import gzip
from nose.plugins.attrib import attr

class AccuracyTest(unittest.TestCase):
    def _get_index(self, f, distance):
        input = 'test/glove.twitter.27B.%dd.txt.gz' % f
        output = 'test/glove.%d.%s.annoy' % (f, distance)
        output_correct = 'test/glove.%d.%s.correct' % (f, distance)
        
        if not os.path.exists(output):
            if not os.path.exists(input):
                # Download GloVe pretrained vectors: http://nlp.stanford.edu/projects/glove/
                # Hosting them on my own S3 bucket since the original files changed format
                url = 'https://s3-us-west-1.amazonaws.com/annoy-vectors/glove.twitter.27B.%dd.txt.gz' % f
                print('downloading', url, '->', input)
                urlretrieve(url, input)

            print('adding items', distance, f)
            annoy = AnnoyIndex(f, distance)
            for i, line in enumerate(gzip.open(input, 'rb')):
                v = [float(x) for x in line.strip().split()[1:]]
                annoy.add_item(i, v);

            print('building index')
            annoy.build(10)
            annoy.save(output)

        annoy = AnnoyIndex(f, distance)
        annoy.load(output)

        if not os.path.exists(output_correct):
            print('finding correct answers')
            f_output = open(output_correct, 'w')
            for i in range(10000):
                js_slow = annoy.get_nns_by_item(i, 11, 100000)[1:]
                assert len(js_slow) == 10
                f_output.write(' '.join(map(str, js_slow)) + '\n')
            f_output.close()

        return annoy, open(output_correct)

    def _test_index(self, f, distance, exp_accuracy):
        annoy, f_correct = self._get_index(f, distance)

        n, k = 0, 0

        for i, line in enumerate(f_correct):
            js_fast = annoy.get_nns_by_item(i, 11, 1000)[1:]
            js_real = [int(x) for x in line.strip().split()]
            assert len(js_fast) == 10
            assert len(js_real) == 10

            n += 10
            k += len(set(js_fast).intersection(js_real))

        accuracy = 100.0 * k / n
        print('%20s %4d accuracy: %5.2f%% (expected %5.2f%%)' % (distance, f, accuracy, exp_accuracy))

        self.assertTrue(accuracy > exp_accuracy - 1.0) # should be within 1%

    def test_angular_25(self):
        self._test_index(25, 'angular', 88.01)

    def test_euclidean_25(self):
        self._test_index(25, 'euclidean', 87.47)

    @attr('slow')
    def test_angular_50(self):
        self._test_index(50, 'angular', 71.67)

    @attr('slow')
    def test_euclidean_50(self):
        self._test_index(50, 'euclidean', 70.28)

    @attr('slow')
    def test_angular_100(self):
        self._test_index(100, 'angular', 53.05)

    @attr('slow')
    def test_euclidean_100(self):
        self._test_index(100, 'euclidean', 56.16)
