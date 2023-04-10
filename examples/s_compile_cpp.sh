#!/bin/bash

for cpp in $(find . -type f -name "*.cpp"); do
    echo "compiling ${cpp}..."
    c++ -Wall "${cpp}" -o $(basename "${cpp}" .cpp) -O2 -DANNOYLIB_MULTITHREADED_BUILD -std=c++14 -pthread -mssse3
    echo "Done"
done