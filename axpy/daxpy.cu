/**
 * Single-precision a * X + B
 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cmath>
#include <ctime>
#include <chrono>
#include "../helper.h"

/**
 * @brief CUDA kernel device code
 * 
 * @param n array length
 * @param a scalar
 * @param x input array
 * @param y input array
 */
__global__ void daxpy(const size_t n, const double a, const double *x, double *y) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

/**
 * Host routine
 */
int main(int argc, char *argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);
    auto err = cudaSuccess;

    // get # of element that fit in memory
    const size_t n = 1 << getSpace(argv[1], sizeof(double), 2);
    const size_t size = n * sizeof(double);
    const double a = 2.0;

    // allocate host vectors
    auto *h_x = (double *) malloc(size);
    auto *h_y = (double *) malloc(size);
    auto *h_z = (double *) malloc(size);

    // verify
    if (h_x == nullptr || h_y == nullptr || h_z == nullptr) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // init
    srand(time(nullptr));
    for (size_t i = 0; i < n; i++) {
        h_x[i] = rand() / (double) RAND_MAX; 
        h_y[i] = rand() / (double) RAND_MAX; 
    }

    // allocate device vectors
    double *d_x, *d_y;
    err = cudaMalloc((void **) &d_x, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector x (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **) &d_y, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector y (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // start time
    auto start = std::chrono::steady_clock::now();

    // cpy host -> device
    err = cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector x from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector y from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // launch kernel
    dim3 threads, blocks;

    threads = dim3(1<<9, 1, 1);
    blocks = dim3(n / threads.x, 1, 1);
    printf("Running daxpy kernel with %d blocks * %d threads\n", blocks.x, threads.x);
    daxpy<<<blocks.x, threads.x>>>(n, a, d_x, d_y);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch daxpy kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // cpy device -> host
    err = cudaMemcpy(h_z, d_y, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector z from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // end time
    auto end = std::chrono::steady_clock::now();
    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    //verify
    double verify;
    for (size_t i = 0; i < n; i++) {
        verify = a * h_x[i] + h_y[i];
        if (fabs(verify - h_z[i]) > 1) {
            fprintf(stderr, "Result verification failed at element %zu: %f != %f * %f + %f - %f!\n", i, verify, a, h_x[i], h_y[i], h_z[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Tetst PASSED\n");

    // free
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    free(h_z);

}