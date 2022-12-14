#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <cstdio>
#include <chrono>
#include <cmath>
#include "../helper.h"

#define MAX_ITER 10000

/**
 * @brief STL implementation to generate a picture of the mandelbrot set
 * 
 * @param in image vector
 * @param width image width
 * @param height image height
 */
void mandelbrot(std::vector<int> &in, const uint64_t width, const uint64_t height) {
    std::for_each(std::execution::par_unseq, 
        in.begin(), in.end(), 
        [ptr = in.data(), num_rows = height, num_cols = width](int& n) { 
            const size_t row = (&n - ptr) / num_cols; 
            const size_t col = (&n - ptr) % num_rows;

            // c = x0 + iy0
            float x0 = ((float)col / num_cols) * 3.5f - 2.5f;
            float y0 = ((float)row / num_rows) * 3.5f - 1.75f;

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
            n = color;
        });
}

int main(int argc, char *argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);

    // memory in GB
    int memory = std::stoi(argv[1]);
    int factor = std::floor(std::sqrt(std::pow(10,9)*memory/sizeof(int))/5000);
    uint64_t height = 5000 * factor,
        width = height,
        n = width * height;

    std::vector<int> x(n, 0);
    printf("Calculating Mandelbrot-Set picture of size %llu x %llu\n", width, height);

    auto start = std::chrono::steady_clock::now();
    mandelbrot(x, width, height);
    auto end = std::chrono::steady_clock::now();

    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // save image
    if (argc > 2 && std::stoi(argv[2]) == 1) {
        FILE* pgmimg;
        pgmimg = fopen("mandelbrot_stl.pgm", "wb");
        fprintf(pgmimg, "P2\n");
        fprintf(pgmimg, "%d %d\n", width, height);
        fprintf(pgmimg, "255\n");
        for (int j = 0; j < height; ++j) {
            for (auto i = 0; i < width; ++i)
                fprintf(pgmimg, "%d ", x[j * height + i]);
            fprintf(pgmimg, "\n");
        }
        fclose(pgmimg);
    }
}
