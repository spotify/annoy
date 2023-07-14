#!/bin/bash

set -e


OPTS="-O2 -DNDEBUG"
#OPTS="-O2 -DNDEBUG -mavx512f -mavx512bw -mavx512dq"

SANITIZER_OPTS="-O1 -g -fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined"

while getopts aA:-: OPT; do
  # support long options: https://stackoverflow.com/a/28466267/519360
  if [ "$OPT" = "-" ]; then   # long option: reformulate OPT and OPTARG
    OPT="${OPTARG%%=*}"       # extract long option name
    OPTARG="${OPTARG#$OPT}"   # extract long option argument (may be empty)
    OPTARG="${OPTARG#=}"      # if long option argument, remove assigning `=`
  fi
  case "$OPT" in
    A | asan )    OPTS="$OPTS $SANITIZER_OPTS" ;;
    a | avx2 )    OPTS="$OPTS -mavx2 -mfma" ;;
    w | avx512 )  OPTS="$OPTS -mavx512f -mavx512bw -mavx512dq" ;;
    ??* )          die "Illegal option --$OPT" ;;  # bad long option
    ? )            exit 2 ;;  # bad short option (error reported via getopts)
  esac
done
shift $((OPTIND-1)) # remove parsed options and args from $@ list

for cpp in $(find . -type f -name "*.cpp"); do
    echo "compiling ${cpp}..."
    c++ -Wall "${cpp}" -o $(basename "${cpp}" .cpp) $OPTS -DANNOYLIB_MULTITHREADED_BUILD -std=c++14 -pthread
    echo "Done"
done