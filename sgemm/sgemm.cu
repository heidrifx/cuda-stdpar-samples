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

/**
 * Single precision general matrix multiplication
 */

#include <cuda_runtime.h>
#include "../helper.h"
#include <cstdio>
#include <cassert>
#include <chrono>

#define TYPE double
#define BLOCK_SIZE (1<<5)

/**
 * @brief CUDA kernel device code for a single precision general matrix multiplication
 * 
 * @tparam T data type
 * @param A input matrix
 * @param B input matrix
 * @param C output matrix
 * @param wA width of matrix A
 * @param wB width of matrix B
 */
template<class T>
__global__ void sgemm(const T *A, const T *B, T *C, const size_t wA, const size_t wB) {
    // block index
    auto bx = blockIdx.x;
    auto by = blockIdx.y;
    // thread index
    auto tx = threadIdx.x;
    auto ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    auto aBegin = wA * BLOCK_SIZE * by;
    // Index of the last sub-matrix of A processed by the block
    auto aEnd = aBegin + wA - 1;
	// Step size used to iterate trough the sub-matrices of A
    auto aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
    auto bBegin = BLOCK_SIZE * bx;
	// Step size used to iterate through the sub-matrices of B
    auto bStep = BLOCK_SIZE;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
    auto Csub = T{};

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
    for (size_t a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// shared memory for the sub-matrices of A and B
        __shared__ T Asub[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ T Bsub[BLOCK_SIZE][BLOCK_SIZE];

		// load matrices from device to shared memory
		// each threads load one element of each matrix
        Asub[ty][tx] = A[a + wA * ty + tx];
        Bsub[ty][tx] = B[b + wB * ty + tx];
        __syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll
        for (size_t k = 0; k < BLOCK_SIZE; ++k) 
            Csub += Asub[ty][k] * Bsub[k][tx];

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
        __syncthreads();
    }

	// Write the block sub-matrix to device memory;
	// each thread writes one element
    auto c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

/**
 * Host routine
 */
int main() {
    size_t block_size = BLOCK_SIZE;

    dim3 dimsA(16 * 8 * block_size, 16 * 8 * block_size, 1),
            dimsB(16 * 8 * block_size, 16 * 8 * block_size, 1);
    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    // allocate host memory for matrices A and B
    size_t size_A = dimsA.x * dimsA.y,
            size_B = dimsB.x * dimsB.y;
    size_t mem_size_A = sizeof(TYPE) * size_A,
            mem_size_B = sizeof(TYPE) * size_B;
    auto *h_A = (TYPE *) malloc(mem_size_A),
            *h_B = (TYPE *) malloc(mem_size_B);

    // verify
    assert(h_A && h_B && "Failed to allocate host memory for matrices A or B!\n");

    // init host memory
    for (size_t i = 0; i < size_A; ++i) {
        h_A[i] = 1.f;
    }
    for (size_t i = 0; i < size_B; ++i) {
        h_B[i] = 0.01f;
    }

    // allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    size_t mem_size_C = dimsC.x * dimsC.y * sizeof(TYPE);
    auto *h_C = (TYPE *) malloc(mem_size_C);

    // verify
    assert(h_C && "Failed to allocate host matrix C!\n");

    // allocate device memory
    TYPE *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **) (&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc((void **) (&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc((void **) (&d_C), mem_size_C));

    // start timer
    auto start = std::chrono::steady_clock::now();

    // cpy host -> device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // launch kernel
    dim3 threads(block_size, block_size),
            grid(dimsB.x / threads.x, dimsA.y / threads.y);

    sgemm<TYPE><<<grid, threads>>>(d_A, d_B, d_C, dimsA.x, dimsB.x);
    checkCudaErrors(cudaGetLastError());

    // cpy device -> host
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    auto end = std::chrono::steady_clock::now();
    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // verify
    bool correct = true;
    double eps = 1.e-6;
    for (int i = 0; i < (int) (dimsC.x * dimsC.y); i++) {
        auto abs_err = fabs(h_C[i] - (dimsA.x * 0.01f));
        auto dot_length = dimsA.x;
        auto abs_val = fabs(h_C[i]);
        auto rel_err = abs_err / abs_val / dot_length;

        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C[i], dimsA.x * 0.01f, eps);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // print result matrix
    /*
    for (size_t i = 0; i < dimsC.x * dimsC.y; i += dimsC.x) {
        printf("(");
        for (size_t j = 0; j < dimsC.x - 1; j++) {
            printf("%f ", h_C[i + j]);
        }
        printf("%f)\n", h_C[i + dimsC.x]);
    }
    */

    // free
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}
