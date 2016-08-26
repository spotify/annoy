import unittest

def execfile(fn):
    with open(fn) as f:
        exec(f.read())

def simple_test():
    execfile('examples/simple_test.py')

def mmap_test():
    execfile('examples/mmap_test.py')

def precision_test():
    execfile('examples/precision_test.py')
