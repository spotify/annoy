#!/bin/bash


echo "compiling precision example..."
for cpp in $(find . -type f -name "*.cpp"); do
    echo "compiling ${cpp}..."
    c++ "${cpp}" -o $(basename "${cpp}" .cpp) -std=c++11 -mssse3
    echo "Done"
done
