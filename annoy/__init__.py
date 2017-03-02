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

from .annoylib import *

class AnnoyIndex(Annoy):
    def __init__(self, f, metric='angular'):
        """
        Initializes an AnnoyIndex that stores vectors of `f` dimensions.

        :param metric: 'angular' or 'euclidean'
        """
        self.f = f
        super(AnnoyIndex, self).__init__(f, metric)

    def check_list(self, vector):
        if type(vector) != list:
            vector = list(vector)
        if len(vector) != self.f:
            raise IndexError('Vector must be of length %d' % self.f)
        return vector

    def check_item(self, item, building=False):
        if item < 0:
            raise IndexError('Item index can not be negative: %d' % item)
        if not building and item >= self.get_n_items():
            raise IndexError('Item index %d is out of range: [0, %d)' %
                (item, self.get_n_items()))
        return item

    def add_item(self, i, vector):
        """
        Adds item `i` (any nonnegative integer) with vector `v`.

        Note that it will allocate memory for `max(i)+1` items.
        """
        # Wrapper to convert inputs to list
        return super(AnnoyIndex, self).add_item(self.check_item(i, building=True), self.check_list(vector))

    def get_nns_by_vector(self, vector, n, search_k=-1, include_distances=False):
        """
        Returns the `n` closest items to vector `vector`.

        :param search_k: the query will inspect up to `search_k` nodes.
        `search_k` gives you a run-time tradeoff between better accuracy and speed.
        `search_k` defaults to `n_trees * n` if not provided.

        :param include_distances: If `True`, this function will return a
        2 element tuple of lists. The first list contains the `n` closest items.
        The second list contains the corresponding distances.
        """
        # Same
        return super(AnnoyIndex, self).get_nns_by_vector(self.check_list(vector), n, search_k, include_distances)

    def get_nns_by_item(self, i, n, search_k=-1, include_distances=False):
        """
        Returns the `n` closest items to item `i`.

        :param search_k: the query will inspect up to `search_k` nodes.
        `search_k` gives you a run-time tradeoff between better accuracy and speed.
        `search_k` defaults to `n_trees * n` if not provided.

        :param include_distances: If `True`, this function will return a
        2 element tuple of lists. The first list contains the `n` closest items.
        The second list contains the corresponding distances.
        """
        # Wrapper to support named arguments
        return super(AnnoyIndex, self).get_nns_by_item(self.check_item(i), n, search_k, include_distances)

    def build(self, n_trees):
        """
        Builds a forest of `n_trees` trees.

        More trees give higher precision when querying. After calling `build`,
        no more items can be added.
        """
        return super(AnnoyIndex, self).build(n_trees)
    
    def unbuild(self):
        """
        Unbuilds the tree in order to allows adding new items.
        build() has to be called again afterwards in order to
        run queries
        """
        return super(AnnoyIndex, self).unbuild()

    def save(self, fn):
        """
        Saves the index to disk.
        """
        return super(AnnoyIndex, self).save(fn)

    def load(self, fn):
        """
        Loads (mmaps) an index from disk.
        """
        return super(AnnoyIndex, self).load(fn)

    def unload(self):
        """
        Unloads an index from disk.
        """
        return super(AnnoyIndex, self).unload()

    def get_item_vector(self, i):
        """
        Returns the vector for item `i` that was previously added.
        """
        return super(AnnoyIndex, self).get_item_vector(self.check_item(i))

    def get_distance(self, i, j):
        """
        Returns the distance between items `i` and `j`.
        """
        return super(AnnoyIndex, self).get_distance(self.check_item(i), self.check_item(j))

    def get_n_items(self):
        """
        Returns the number of items in the index.
        """
        return super(AnnoyIndex, self).get_n_items()

    def set_seed(self, seed):
        """
        Sets the seed of Annoy's random number generator.
        """
        return super(AnnoyIndex, self).set_seed(seed)
