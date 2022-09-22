#!/bin/bash
man="\
Usage: compile_local [use_nvc++_for_cuda]\n\
This will compile all CUDA Kernels (.cu) and all .cpp files using Parallel STL\n \
   use_nvc++_for_cuda\t set to 1 if the CUDA kernels should be compiled with nvc++\n\
"

# -h message
if [[ "$1" == "-h" || "$1" == "--help" ]]; then printf "$man"; exit; fi

compiler=$([ "$1" == 1 ] && echo nvc++ || echo nvcc)

mkdir -p bin/

# then compile CUDA
find **/*.cu -exec sh -c '$1 {} -o bin/$(basename {} .cu)' sh $compiler \;

# after that compile STL
find **/*.cpp -exec sh -c 'nvc++ -stdpar {} -o bin/$(basename {} .cpp)' \;
