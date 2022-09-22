#include <vector>
#include <algorithm>
#include <execution>
#include <random>
#include <ctime>
#include <chrono>
#include "../helper.h"

/**
 * @brief STL implementation of saxpy - in-place
 * 
 * @param a scalar
 * @param x input vector
 * @param y input vector
 */
void saxpy(const float a, const std::vector<float> &x, std::vector<float> &y) {
    std::transform(std::execution::par_unseq, x.begin(), x.end(), y.begin(), y.begin(),
                   [=](auto x, auto y) { return y + a * x; });
}

int main(int argc, char *argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);

    // get # of element that fit in memory
    int numElements = 1 << getSpace(argv[1], sizeof(float), 2);
    printf("Number of elements %d\n", numElements);

    // input and tmp vector
    std::vector<float> x(numElements), y(numElements), tmp(numElements);
    float a = 0.3f;

    // init
    srand(time(NULL));
    for (int i = 0; i < numElements; ++i) {
        x[i] = rand() / (float) RAND_MAX;
        y[i] = rand() / (float) RAND_MAX;
        tmp[i] = y[i]; // save for verify as saxpy is in-place and will override y
    }

    auto start = std::chrono::steady_clock::now();
    saxpy(a, x, y);
    auto end = std::chrono::steady_clock::now();

    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // verify
    for (size_t i = 0; i < numElements; ++i) {
        auto verify = a * x[i] + tmp[i] - y[i];
        if (fabs(verify > 1e-5)) {
            fprintf(stderr, "Result verification failed at element %zu!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED!\n");
}
