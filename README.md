# Comparison of CUDA and stdpar samples

This repo includes all algorithms and problems used in the bachelor thesis "*Comparison of Parallel STL implementations for GPUs with native frameworks*" (original title: "*Vergleich von Parallel STL Implementierungen für GPUs mit nativen Frameworks*").

## Samples

Each sample is implemented in CUDA and stdpar and all are contained in seperate folders.

- Vector addition
  
  - also available in *CUDA with USM* and a *multi* version which does multiple runs using different input sizes

- `axpy` (with single- and double-percision)

- Vector reduction
  
  - includes two different approaches in CUDA, a naïve and a more optimized implementation

- Filter-Map-Reduce algorithm
  
  - includes two different approaches on stdpar, a single-call implementation and one with three STL calls

- General matrix multiplication

- Mandelbrot set
  
  - also available in *CUDA with USM* and a *multi* version which does multiple runs using different input sizes

## Compilation and usage

### Requirements

- CUDA 11.4+

- NVHPC SDK 22.5+

Older versions might work as well but weren't tested.

### Usage

- Compile the samples using `compile_local.sh [use_nvc++_for_cuda]`. The executables will be saved in the `/bin` directory.
  
  - `use_nvc++_for_cuda` is optional and if `1` passed as paramter `nvc++` will be used for all samples.

- Aggregate data using `aggregateData_local.sh memory_size repeats [clear_data] [executable]`. The result will be saved in the `/data` directory.
  
  - `memory_size` determines the memory size in GB used on the GPU.
  
  - `repeats` determines the amount of repeats that should be done.
  
  - `clear_data` is *optional* and will clear previously aggregated data stored in `/data`.
  
  - `executable` is optional and allows for an executable or directory to be used as samples instead of the `/bin` directory.

Both scripts support `-h` and `--help` as options to display a short usage message. The scripts `compile.sh` and `aggregateData.sh` were used on other clusters.
