#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <iostream>
#include "../helper.h"
#include <chrono>

#define MAX_ITER 10000

/**
 * @brief CUDA kernel device code to generate a picture of the mandelbrot set
 * 
 * @param pos image array
 * @param width image width
 * @param height  image height
 */
__global__ void calc(int *pos, const uint64_t width, const uint64_t height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // WIDTH
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // HEIGHT
    uint64_t idx = row * width + col,
        n = width * height;

    // return if current index, col or row is out of bounds
    if(col >= width || row >= height || idx >= n) return;

    // c = x0 + iy0
    float x0 = ((float)col / width) * 3.5f - 2.5f;
    float y0 = ((float)row / height) * 3.5f - 1.75f;

    // z = x + iy
    float x = 0.0f;
    float y = 0.0f;
    int iter = 0;
    float xtemp;

    while((x * x + y * y <= 4.0f) && (iter < MAX_ITER)) { 
        // z^2 = x^2 + i2xy - y^2
        // Re(z^2 + c) = x^2 - y^2 +x0
        xtemp = x * x - y * y + x0;

        // Im(z^2 + c) = 2xy + y0
        y = 2.0f * x * y + y0;

        x = xtemp;
        iter++;
    }

    int color = iter * 5;
    if (color >= 256) color = 0;
    pos[idx] = color;
}

void mandelbrot(const int factor) {
    uint64_t height = 5000 * factor,
        width = height,
        n = width * height;

    int* image_buffer;
    printf("Calculating Mandelbrot-Set picture of size %llu x %llu\n", width, height);

    checkCudaErrors(cudaMallocManaged(&image_buffer, sizeof(int) * n));

    // start timer
    auto start = std::chrono::steady_clock::now();

    dim3 block_size(16, 16);
    dim3 grid_size(width / block_size.x, height / block_size.y);
    calc<<<grid_size, block_size>>>(image_buffer, width, height);
    cudaDeviceSynchronize();

    // stop timer
    auto end = std::chrono::steady_clock::now();
    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    checkCudaErrors(cudaFree(image_buffer));
}

/**
 * Host routine
 */
int main() {
    // run until 45,000 pixel image width is reached
    for(int i = 1; i <= 9; ++i)
        mandelbrot(i);
}
