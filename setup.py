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
import codecs
import os
import sys

readme_note = """\
.. note::

   For the latest source, discussion, etc, please visit the
   `GitHub repository <https://github.com/spotify/annoy>`_\n\n

.. image:: https://img.shields.io/github/stars/spotify/annoy.svg
    :target: https://github.com/spotify/annoy

"""

with codecs.open('README.rst', encoding='utf-8') as fobj:
    long_description = readme_note + fobj.read()

if os.environ.get('TRAVIS') == 'true':
    # Resolving some annoying issue
    travis_extra_compile_args = ['-mno-avx']
else:
    travis_extra_compile_args = []

# Not all CPUs have march as a tuning parameter
import platform
cputune = ['-march=native',]
if platform.machine() == "ppc64le":
    cputune = ['-mcpu=native',]

if os.name != 'nt':
    compile_args = ['-O3', '-ffast-math', '-fno-associative-math']
else:
    compile_args = []
    cputune = []

setup(name='annoy',
      version='1.10.0',
      description='Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk.',
      packages=['annoy'],
      ext_modules=[
        Extension(
            'annoy.annoylib', ['src/annoymodule.cc'],
            depends=['src/annoylib.h', 'src/kissrandom.h', 'src/mman.h'],
            extra_compile_args=compile_args + cputune + travis_extra_compile_args
        )
      ],
      long_description=long_description,
      author='Erik Bernhardsson',
      author_email='mail@erikbern.com',
      url='https://github.com/spotify/annoy',
      license='Apache License 2.0',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],
      keywords='nns, approximate nearest neighbor search',
      setup_requires=['nose>=1.0']
    )
