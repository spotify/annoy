#!/usr/bin/env python
# Copyright (c) 2013 Spotify AB
# -*- coding: utf-8 -*-

# from spotify.build import setup
from distutils.core import setup, Extension
import os

setup(name='annoy',
      version='1.0',
      description='Approximate nearest neighbor',
      packages=['annoy'],
      ext_modules=[Extension('annoy.annoylib', ['src/annoylib.cc'], libraries=['boost_python'])],
    )
