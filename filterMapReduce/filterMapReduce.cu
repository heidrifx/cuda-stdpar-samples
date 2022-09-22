/**
 * Filter even numbers - map * 2 - reduce add up
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>
#include "../helper.h"
#include <chrono>

#define N (1 << getSpace(argv[1], sizeof(int), 2))
#define BLOCK_SIZE (1 << 9)
#define RANDOM 1

/**
 * CUDA Kernel Device code
 */
__global__ void fmr(const int *g_inputData, int *g_outputData) {
    extern __shared__ int sharedData[];
    auto tid = threadIdx.x,
            i = blockIdx.x * blockDim.x + threadIdx.x;

    // copy to shared memory
    sharedData[tid] = g_inputData[i];
    // wait for all to finish
    __syncthreads();

    // filter odd numbers out (i.e. set them to neutral element of addition 0)
    // map all even numbers to double their amount
    if (tid < blockDim.x && sharedData[tid] > 0) {
        (sharedData[tid] & 1) ? (sharedData[tid] = 0) : (sharedData[tid] *= 2);
    }
    __syncthreads();

    // reduce
    for (auto s = 1; s < blockDim.x; s *= 2) {
        auto index = 2 * s * tid;
        if (index < blockDim.x) sharedData[index] += sharedData[index + s];
        __syncthreads();
    }

    if (tid == 0) g_outputData[blockIdx.x] = sharedData[0];
}

/**
 * Host routine
 */
int main(int argc, char *argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);
    size_t size_V = N,
            size_R = (size_V + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto mem_size_V = sizeof(int) * size_V,
            mem_size_R = sizeof(int) * size_R;

    // allocate host memory for vectors
    auto *h_V = (int *) malloc(mem_size_V),
            *h_R = (int *) malloc(mem_size_R);

    // verify
    assert(h_V && h_R && "Failed to allocate host memory for vectors!\n");

    // init host input vector
#if RANDOM
    srand(time(nullptr));
    for (auto i = 0; i < size_V; ++i) h_V[i] = rand() % 20;
#else
    h_V[0] = 12;
    for (auto i = 1; i < size_V; i++) h_V[i] = h_V[i - 1] + 1;
#endif

    // print some values
    printf("( ");
    for (auto i = 0; i < std::min(10, N); i++) {
        printf("%d ", h_V[i]);
    }
    printf("%s)\n", N > 10 ? "..." : "");

    // allocate device memory
    int *d_V, *d_R;
    checkCudaErrors(cudaMalloc((void **) (&d_V), mem_size_V));
    checkCudaErrors(cudaMalloc((void **) (&d_R), mem_size_R));

    // start timer
    auto start = std::chrono::steady_clock::now();

    // cpy host -> device
    checkCudaErrors(cudaMemcpy(d_V, h_V, mem_size_V, cudaMemcpyHostToDevice));

    // launch kernel
    uint threadsPerBlock = BLOCK_SIZE,
            blocksPerGrid = (size_V + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    fmr<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(d_V, d_R);
    checkCudaErrors(cudaGetLastError());

    // cpy device -> host
    checkCudaErrors(cudaMemcpy(h_R, d_R, mem_size_R, cudaMemcpyDeviceToHost));

    auto end = std::chrono::steady_clock::now();
    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // print tmp result
    /*
    printf("intermediate result: ( ");
    for (auto i = 0; i < std::min(10, N); i++) {
        printf("%d ", h_Tmp[i]);
    }
    printf("%s)\n", N > 10 ? "..." : "");
    */

    // print result
    int result = 0;
    for (auto i = 0; i < size_R; i++) {
        result += h_R[i];
    }
    printf("result: %d\n", result);

    auto tmp = 0;
    for (int i = 0; i < size_V; ++i)
        tmp += (h_V[i] & 1) ? 0 : h_V[i] * 2;

    if (abs(result - tmp) > 0) {
        fprintf(stderr, "Result verification failed at element!\n");
        exit(EXIT_FAILURE);
    }
    printf("Test PASSED!\n");

    // free
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_R));
    free(h_V);
    free(h_R);

}