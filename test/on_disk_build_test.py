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

import pytest

from annoy import AnnoyIndex


@pytest.fixture(scope="module", autouse=True)
def setUp():
    if os.path.exists("on_disk.ann"):
        os.remove("on_disk.ann")


def add_items(i):
    i.add_item(0, [2, 2])
    i.add_item(1, [3, 2])
    i.add_item(2, [3, 3])


def check_nns(i):
    assert i.get_nns_by_vector([4, 4], 3) == [2, 1, 0]
    assert i.get_nns_by_vector([1, 1], 3) == [0, 1, 2]
    assert i.get_nns_by_vector([4, 2], 3) == [1, 2, 0]


def test_on_disk():
    f = 2
    i = AnnoyIndex(f, "euclidean")
    i.on_disk_build("on_disk.ann")
    add_items(i)
    i.build(10)
    check_nns(i)
    i.unload()
    i.load("on_disk.ann")
    check_nns(i)
    j = AnnoyIndex(f, "euclidean")
    j.load("on_disk.ann")
    check_nns(j)
