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

from setuptools import setup, Extension
import os
import sys

readme_note = """\
.. note::

   For the latest source, discussion, etc, please visit the
   `GitHub repository <https://github.com/spotify/annoy>`_\n\n
"""

with open('README.rst') as fobj:
    long_description = readme_note + fobj.read()


def get_libraries():
    if sys.version_info < (3,):
        return ['boost_python']
    else:
        return ['boost_python3']

setup(name='annoy',
      version='1.0.5',
      description='Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk.',
      packages=['annoy'],
      ext_modules=[Extension('annoy.annoylib', ['src/annoymodule.cc'], libraries=get_libraries())],
      long_description=long_description,
      author='Erik Bernhardsson',
      author_email='erikbern@spotify.com',
      url='https://github.com/spotify/annoy',
      license='Apache License 2.0',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
      ],
      keywords='nns, approximate nearest neighbor search',
      setup_requires=['nose>=1.0']
    )
