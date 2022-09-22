#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <cstdio>
#include <chrono>
#include "../helper.h"

#define TYPE int //long

template<class T>
T fmr(const std::vector<T> &x) {
    std::vector<T> tmp(x.size());
    // filter
    std::transform(std::execution::par_unseq, x.begin(), x.end(), tmp.begin(), [=](auto x) { return x & 1 ? 0 : x; });
    // map
    std::transform(std::execution::par_unseq, tmp.begin(), tmp.end(), tmp.begin(), [=](auto x) { return x * 2; });
    // reduce
    return std::reduce(std::execution::par_unseq, tmp.begin(), tmp.end(), 0);
}

template<class T>
T optimized_fmr(const std::vector<T> &x) {
    return std::transform_reduce(std::execution::par_unseq, x.begin(), x.end(), 0, [=](auto x, auto y) { return x + y; },
                          [=](auto x) { return x & 1 ? 0 : x * 2; });
}

int main(int argc, char *argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);
    int numElements = 1 << getSpace(argv[1], sizeof(TYPE), 2);
    printf("Number of elements %d\n", numElements);

    std::vector<TYPE> x(numElements);

    srand(time(nullptr));
    for (int i = 0; i < numElements; ++i) {
        x[i] = rand() % 200;
    }

    auto start = std::chrono::steady_clock::now();
    auto mappedSum = optimized_fmr(x);
    auto end = std::chrono::steady_clock::now();

    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    auto tmp = 0;
    for (int i = 0; i < numElements; ++i)
        tmp += (x[i] & 1) ? 0 : x[i] * 2;

    if (abs(mappedSum - tmp) > 0) {
        fprintf(stderr, "Result verification failed at element!\n");
        exit(EXIT_FAILURE);
    }
    printf("Test PASSED!\n");
}
