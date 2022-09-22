#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <cstdio>
#include <chrono>
#include <cmath>
#include "../helper.h"

#define MAX_ITER 10000

void mandelbrot(std::vector<int> &in, const ull_int width, const ull_int height) {
    std::for_each(std::execution::par_unseq, 
        in.begin(), in.end(), 
        [ptr = in.data(), num_rows = height, num_cols = width](int& n) { 
            const size_t row = (&n - ptr) / num_cols; 
            const size_t col = (&n - ptr) % num_rows;

            float x0 = ((float)col / num_cols) * 3.5f - 2.5f;
            float y0 = ((float)row / num_rows) * 3.5f - 1.75f;

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
            n = color;
        });
}

void mandelbrot(const int factor) {
    ull_int height = 5000 * factor,
        width = height,
        n = width * height;

    std::vector<int> x(n, 0);
    printf("Calculating Mandelbrot-Set picture of size %llu x %llu\n", width, height);

    auto start = std::chrono::steady_clock::now();
    mandelbrot(x, width, height);
    auto end = std::chrono::steady_clock::now();

    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}

int main(int argc, char *argv[]) {
    int cpu = (argc > 1 && std::stoi(argv[1]) == 1) ? 3 : 9;
    printf("cpu? %d\n", cpu);
    for(int i = 1; i <= cpu; ++i)
        mandelbrot(i);
}
