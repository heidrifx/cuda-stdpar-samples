#!/bin/bash
# first load all modules
module load cuda/11.4.3 /data/scratch/heidrifx/nvidia/hpc_sdk/modulefiles/nvhpc/22.5
# then compile CUDA
~/fd/fd --max-depth=1 -x nvcc {} -o bin/{.} \; \.cu
# after that compile STL
~/fd/fd --max-depth=1 -x nvc++ -stdpar {} -o bin/{.} \; \.cpp

