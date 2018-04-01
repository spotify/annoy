#!/bin/bash


echo "compiling precision example..."
cmd="g++ precision_test.cpp -o precision_test -std=c++11 -pthread"
eval $cmd
echo "Done"
