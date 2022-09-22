/**
 * Compute the mandelbrot set
 */

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include "../helper.h"
#include <chrono>

#define MAX_ITER 10000

/**
 * CUDA Kernel Device code
 */
__global__ void calc(int *pos, const ull_int width, const ull_int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // WIDTH
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // HEIGHT
    ull_int idx = row * width + col,
        n = width * height;

    if(col >= width || row >= height || idx >= n) return;

    float x0 = ((float)col / width) * 3.5f - 2.5f;
    float y0 = ((float)row / height) * 3.5f - 1.75f;

    float x = 0.0f;
    float y = 0.0f;
    int iter = 0;
    float xtemp;
    while((x * x + y * y <= 4.0f) && (iter < MAX_ITER)) { 
        xtemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xtemp;
        iter++;
    }

    int color = iter * 5;
    if (color >= 256) color = 0;
    pos[idx] = color;
}

/**
 * Host routine
 */
int main(int argc, char *argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);

    int memory = std::stoi(argv[1]);
    int factor = floor(std::sqrt(pow(10,9)*memory/sizeof(int))/5000);
    ull_int height = 5000 * factor,
        width = height,
        n = width * height;

    int* d_image_buffer;
    auto *h_image_buffer = (int *) malloc(sizeof(int) * n);
    printf("Calculating Mandelbrot-Set picture of size %llu x %llu\n", width, height);

    // start timer
    auto start = std::chrono::steady_clock::now();

    checkCudaErrors(cudaMalloc(&d_image_buffer, sizeof(int) * n));

    dim3 block_size(16, 16);
    dim3 grid_size(width / block_size.x, height / block_size.y);
    calc<<<grid_size, block_size>>>(d_image_buffer, width, height);
    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemcpy(h_image_buffer, d_image_buffer, sizeof(int) * n, cudaMemcpyDeviceToHost));

    // stop timer
    auto end = std::chrono::steady_clock::now();
    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    checkCudaErrors(cudaFree(d_image_buffer));

    if (argc > 2 && std::stoi(argv[2]) == 1) {
        FILE* pgmimg;
        pgmimg = fopen("mandelbrot_cuda.pgm", "wb");
        fprintf(pgmimg, "P2\n");
        fprintf(pgmimg, "%d %d\n", width, height);
        fprintf(pgmimg, "255\n");
        for (int j = 0; j < height; ++j) {
            for (auto i = 0; i < width; ++i)
                fprintf(pgmimg, "%d ", h_image_buffer[j * height + i]);
            fprintf(pgmimg, "\n");
        }
        fclose(pgmimg);
    }

    free(h_image_buffer);
}
