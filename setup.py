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
import platform
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

# Various platform-dependent extras
extra_compile_args = ['-D_CRT_SECURE_NO_WARNINGS', '-fpermissive']
extra_link_args = []

# Not all CPUs have march as a tuning parameter
cputune = ['-march=native',]
if platform.machine() == 'ppc64le':
    extra_compile_args += ['-mcpu=native',]

if platform.machine() == 'x86_64':
    extra_compile_args += cputune

if os.name != 'nt':
    extra_compile_args += ['-O3', '-ffast-math', '-fno-associative-math']

# Add multithreaded build flag for all platforms using Python 3 and
# for non-Windows Python 2 platforms
python_major_version = sys.version_info[0]
if python_major_version == 3 or (python_major_version == 2 and os.name != 'nt'):
    extra_compile_args += ['-DANNOYLIB_MULTITHREADED_BUILD']

    if os.name != 'nt':
        extra_compile_args += ['-std=c++14']

# #349: something with OS X Mojave causes libstd not to be found
if platform.system() == 'Darwin':
    extra_compile_args += ['-mmacosx-version-min=10.12']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.12']

# Manual configuration, you're on your own here.
manual_compiler_args = os.environ.get('ANNOY_COMPILER_ARGS', None)
if manual_compiler_args:
    extra_compile_args = manual_compiler_args.split(',')
manual_linker_args = os.environ.get('ANNOY_LINKER_ARGS', None)
if manual_linker_args:
    extra_link_args = manual_linker_args.split(',')

setup(name='annoy',
      version='1.17.1',
      description='Approximate Nearest Neighbors in C++/Python optimized for memory usage and loading/saving to disk.',
      packages=['annoy'],
      package_data={'annoy': ['__init__.pyi', 'py.typed']},
      ext_modules=[
          Extension(
              'annoy.annoylib', ['src/annoymodule.cc'],
              depends=['src/annoylib.h', 'src/kissrandom.h', 'src/mman.h'],
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
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
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],
      keywords='nns, approximate nearest neighbor search',
      setup_requires=['nose>=1.0'],
      tests_require=['numpy', 'h5py']
      )
