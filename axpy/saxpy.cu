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
 * CUDA Kernel Device code
 */
__global__ void saxpy(const int n, const float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

/**
 * Host routine
 */
int main(int argc, char *argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);
    auto err = cudaSuccess;

    const size_t n = 1 << getSpace(argv[1], sizeof(float), 2);
    const size_t size = n * sizeof(float);
    const float a = 2.f;

    // allocate host vectors
    auto *h_x = (float *) malloc(size);
    auto *h_y = (float *) malloc(size);
    auto *h_z = (float *) malloc(size);

    // verify
    if (h_x == nullptr || h_y == nullptr || h_z == nullptr) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // init
    srand(time(nullptr));
    for (size_t i = 0; i < n; i++) {
        h_x[i] = 1.f / (rand() % 200 + 1); 
        h_y[i] = 1.f / (rand() % 200 + 1); 
    }

    // allocate device vectors
    float *d_x, *d_y;
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
    saxpy<<<blocks, threads>>>(n, a, d_x, d_y);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch saxpy kernel (error code %s)!\n",
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
    float verify;
    for (size_t i = 0; i < n; i++) {
        verify = a * h_x[i] + h_y[i] - h_z[i];
        if (fabs(verify) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %zu!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    // free
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    free(h_z);

}