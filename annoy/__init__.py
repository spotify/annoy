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
        :param metric: 'angular' or 'euclidean'
        """
        super(AnnoyIndex, self).__init__(f, metric)

    def add_item(self, i, vector):
        # Wrapper to convert inputs to list
        return super(AnnoyIndex, self).add_item(i, list(vector))

    def get_nns_by_vector(self, vector, n, search_k=-1):
        # Same
        return super(AnnoyIndex, self).get_nns_by_vector(list(vector), n, search_k)
