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
import random

import pytest

from annoy import AnnoyIndex


def test_not_found_tree():
    i = AnnoyIndex(10, "angular")
    with pytest.raises(IOError):
        i.load("nonexists.tree")


def test_binary_compatibility():
    i = AnnoyIndex(10, "angular")
    i.load("test/test.tree")

    # This might change in the future if we change the search algorithm, but in that case let's update the test
    assert i.get_nns_by_item(0, 10) == [0, 85, 42, 11, 54, 38, 53, 66, 19, 31]


def test_load_unload():
    # Issue #108
    i = AnnoyIndex(10, "angular")
    for x in range(100000):
        i.load("test/test.tree")
        i.unload()


def test_construct_load_destruct():
    for x in range(100000):
        i = AnnoyIndex(10, "angular")
        i.load("test/test.tree")


def test_construct_destruct():
    for x in range(100000):
        i = AnnoyIndex(10, "angular")
        i.add_item(1000, [random.gauss(0, 1) for z in range(10)])


def test_save_twice():
    # Issue #100
    t = AnnoyIndex(10, "angular")
    for i in range(100):
        t.add_item(i, [random.gauss(0, 1) for z in range(10)])
    t.build(10)
    t.save("t1.ann")
    t.save("t2.ann")


def test_load_save():
    # Issue #61
    i = AnnoyIndex(10, "angular")
    i.load("test/test.tree")
    u = i.get_item_vector(99)
    i.save("i.tree")
    v = i.get_item_vector(99)
    assert u == v
    j = AnnoyIndex(10, "angular")
    j.load("test/test.tree")
    w = i.get_item_vector(99)
    assert u == w
    # Ensure specifying if prefault is allowed does not impact result
    j.save("j.tree", True)
    k = AnnoyIndex(10, "angular")
    k.load("j.tree", True)
    x = k.get_item_vector(99)
    assert u == x
    k.save("k.tree", False)
    l = AnnoyIndex(10, "angular")
    l.load("k.tree", False)
    y = l.get_item_vector(99)
    assert u == y


def test_save_without_build():
    t = AnnoyIndex(10, "angular")
    for i in range(100):
        t.add_item(i, [random.gauss(0, 1) for z in range(10)])
    # Note: in earlier version, this was allowed (see eg #61)
    with pytest.raises(Exception):
        t.save("x.tree")


def test_unbuild_with_loaded_tree():
    i = AnnoyIndex(10, "angular")
    i.load("test/test.tree")
    with pytest.raises(Exception):
        i.unbuild()


def test_seed():
    i = AnnoyIndex(10, "angular")
    i.load("test/test.tree")
    i.set_seed(42)


def test_unknown_distance():
    with pytest.raises(Exception):
        AnnoyIndex(10, "banana")


def test_metric_kwarg():
    # Issue 211
    i = AnnoyIndex(2, metric="euclidean")
    i.add_item(0, [1, 0])
    i.add_item(1, [9, 0])
    assert i.get_distance(0, 1) == pytest.approx(8)
    assert i.f == 2


def test_metric_f_kwargs():
    AnnoyIndex(f=3, metric="euclidean")


def test_item_vector_after_save():
    # Issue #279
    a = AnnoyIndex(3, "angular")
    a.verbose(True)
    a.add_item(1, [1, 0, 0])
    a.add_item(2, [0, 1, 0])
    a.add_item(3, [0, 0, 1])
    a.build(-1)
    assert a.get_n_items() == 4
    assert a.get_item_vector(3) == [0, 0, 1]
    assert set(a.get_nns_by_item(1, 999)) == set([1, 2, 3])
    a.save("something.annoy")
    assert a.get_n_items() == 4
    assert a.get_item_vector(3) == [0, 0, 1]
    assert set(a.get_nns_by_item(1, 999)) == set([1, 2, 3])


def test_prefault():
    i = AnnoyIndex(10, "angular")
    i.load("test/test.tree", prefault=True)
    assert i.get_nns_by_item(0, 10) == [0, 85, 42, 11, 54, 38, 53, 66, 19, 31]


def test_fail_save():
    t = AnnoyIndex(40, "angular")
    with pytest.raises(IOError):
        t.save("")


def test_overwrite_index():
    # Issue #335
    f = 40

    # Build the initial index
    t = AnnoyIndex(f, "angular")
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(f)]
        t.add_item(i, v)
    t.build(10)
    t.save("test.ann")

    # Load index file
    t2 = AnnoyIndex(f, "angular")
    t2.load("test.ann")

    # Overwrite index file
    t3 = AnnoyIndex(f, "angular")
    for i in range(500):
        v = [random.gauss(0, 1) for z in range(f)]
        t3.add_item(i, v)
    t3.build(10)
    if os.name == "nt":
        # Can't overwrite on Windows
        with pytest.raises(IOError):
            t3.save("test.ann")
    else:
        t3.save("test.ann")
        # Get nearest neighbors
        v = [random.gauss(0, 1) for z in range(f)]
        t2.get_nns_by_vector(v, 1000)  # Should not crash


def test_get_n_trees():
    i = AnnoyIndex(10, "angular")
    i.load("test/test.tree")
    assert i.get_n_trees() == 10


def test_write_failed():
    f = 40

    # Build the initial index
    t = AnnoyIndex(f, "angular")
    t.verbose(True)
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(f)]
        t.add_item(i, v)
    t.build(10)

    if os.name == "nt":
        path = "Z:\\xyz.annoy"
    else:
        path = "/x/y/z.annoy"
    with pytest.raises(Exception):
        t.save(path)


def test_dimension_mismatch():
    t = AnnoyIndex(100, "angular")
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for z in range(100)])
    t.build(10)
    t.save("test.annoy")

    u = AnnoyIndex(200, "angular")
    with pytest.raises(IOError):
        u.load("test.annoy")
    u = AnnoyIndex(50, "angular")
    with pytest.raises(IOError):
        u.load("test.annoy")


def test_add_after_save():
    # 398
    t = AnnoyIndex(100, "angular")
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for z in range(100)])
    t.build(10)
    t.save("test.annoy")

    # Used to segfault:
    v = [random.gauss(0, 1) for z in range(100)]
    with pytest.raises(Exception):
        t.add_item(i, v)


def test_build_twice():
    # 420
    t = AnnoyIndex(100, "angular")
    for i in range(1000):
        t.add_item(i, [random.gauss(0, 1) for z in range(100)])
    t.build(10)
    # Used to segfault:
    with pytest.raises(Exception):
        t.build(10)


def test_very_large_index():
    # 388
    f = 3
    dangerous_size = 2**31
    size_per_vector = 4 * (f + 3)
    n_vectors = int(dangerous_size / size_per_vector)
    m = AnnoyIndex(3, "angular")
    m.verbose(True)
    for i in range(100):
        m.add_item(n_vectors + i, [random.gauss(0, 1) for z in range(f)])
    n_trees = 10
    m.build(n_trees)
    path = "test_big.annoy"
    m.save(path)  # Raises on Windows

    # Sanity check size of index
    assert os.path.getsize(path) >= dangerous_size
    assert os.path.getsize(path) < dangerous_size + 100e3

    # Sanity check number of trees
    assert m.get_n_trees() == n_trees
