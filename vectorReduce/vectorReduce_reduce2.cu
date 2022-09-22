/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cuda_runtime.h>
#include "../helper.h"
#include <cstdio>
#include <cassert>
#include <ctime>
#include <algorithm>
#include <chrono>

#define TYPE int
#define RANDOM 1
#define N (1 << getSpace(argv[1], sizeof(TYPE), 2))
#define BLOCK_SIZE (1 << 9)

/**
 * @brief CUDA kernel device code for a simple vector reduction
 * 
 * @tparam T data type
 * @param g_idata input array
 * @param g_odata output array
 * @param n input length
 */
template<class T>
__global__ void reduce2(const T *g_idata, T *g_odata, const uint64_t n) {
    // shared memory
    extern __shared__ T sdata0[];
    // threadID & array index
    auto tid = threadIdx.x,
            i = blockIdx.x * blockDim.x + threadIdx.x;

    // check if inside array
    if (i < n) {
        // copy to shared memory
        sdata0[tid] = g_idata[i];
        __syncthreads();

        // reduce
        for (auto s = 1; s < blockDim.x; s *= 2) {
            auto index = 2 * s * tid;
            if (index < blockDim.x) sdata0[index] += sdata0[index + s];
            __syncthreads();
        }

        // copy result from shared to global memory
        if (tid == 0) g_odata[blockIdx.x] = sdata0[0];
    }
}

/**
 * Host routine
 */
int main(int argc, char *argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);
    size_t size_V = N,
            size_R = (size_V + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto mem_size_V = sizeof(TYPE) * size_V,
            mem_size_R = sizeof(TYPE) * size_R;

    // allocate host memory for vectors;
    auto *h_V = (TYPE *) malloc(mem_size_V),
            *h_R = (TYPE *) malloc(mem_size_R);

    // verify
    assert(h_V && h_R && "Failed to allocate host memory for vectors!\n");

    // init host input vector
    h_V[0] = 0;

#if RANDOM
    srand(time(nullptr));
    for (size_t i = 0; i < size_V; i++)
        h_V[i] = rand() % 200; // random int
#else
    for (size_t i = 1; i < size_V; i++)
        h_V[i] = h_V[i - 1] + 1; // increasing numbers
#endif

    // print some values
    printf("( ");
    for (size_t i = 0; i < std::min(5, N); i++)
        printf("%d ", h_V[i]);
    printf("%s)\n", N > 5 ? "..." : "");

    // allocate device memory
    TYPE *d_V, *d_R;
    checkCudaErrors(cudaMalloc((void **) (&d_V), mem_size_V));
    checkCudaErrors(cudaMalloc((void **) (&d_R), mem_size_R));

    // start timer
    auto start = std::chrono::steady_clock::now();

    // cpy host -> device
    checkCudaErrors(cudaMemcpy(d_V, h_V, mem_size_V, cudaMemcpyHostToDevice));

    // launch kernel
    dim3 threads, blocks;

    threads = dim3(1<<9, 1, 1);
    blocks = dim3(size_V / threads.x, 1, 1);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocks.x, threads.x);
    reduce2<TYPE><<<blocks, threads, threads.x * sizeof(TYPE)>>>(d_V, d_R, N);
    checkCudaErrors(cudaGetLastError());

    // cpy device -> host
    checkCudaErrors(cudaMemcpy(h_R, d_R, mem_size_R, cudaMemcpyDeviceToHost));

    auto end = std::chrono::steady_clock::now();
    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // print result
    TYPE result = 0.f;
    for (size_t i = 0; i < size_R - 1; i++) {
        result += h_R[i];
    }
    result += h_R[size_R - 1];

    // verify
    TYPE verify = 0, diff = 0;
#if RANDOM
    for (size_t i = 0; i < size_V; i++) verify += h_V[i];
#else
    TYPE n = N - 1;
    verify = (n * n + n) / 2;
#endif
    diff = abs(verify - result);
    printf("Verify: %d, Diff: %d\n", verify, diff);
    printf("%s\n", diff < 1e-4 ? "Result = PASS" : "Result = FAIL");

    // free
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_R));
    free(h_V);
    free(h_R);
}
