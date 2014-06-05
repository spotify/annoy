#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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

from distutils.core import setup, Extension
import os

long_description = ['Note: For the latest source, discussion, etc, please visit the `Github repository <https://github.com/spotify/annoy>`_\n\n']
for line in open('README.rst'):
    long_description.append(line)
long_description = ''.join(long_description)

setup(name='annoy',
      version='1.0.3',
      description='Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk.',
      packages=['annoy'],
      ext_modules=[Extension('annoy.annoylib', ['src/annoylib.cc'], libraries=['boost_python'])],
      long_description=long_description,
      author='Erik Bernhardsson',
      author_email='erikbern@spotify.com',
      url='https://github.com/spotify/luigi',
      license='Apache License 2.0',
    )
