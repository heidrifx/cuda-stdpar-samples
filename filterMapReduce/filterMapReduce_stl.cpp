#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <cstdio>
#include <chrono>
#include "../helper.h"

#define TYPE int //long

/**
 * @brief Naive STL implementation of a filter-map-reduce operation
 * 
 * @tparam T data type
 * @param x input vector
 * @return T filtered, mapped and reduced vector
 */
template<class T>
T fmr(std::vector<T> &x) {
    // filter
    std::transform(std::execution::par_unseq, x.begin(), x.end(), x.begin(), [=](auto x) { return x & 1 ? 0 : x; });
    // map
    std::transform(std::execution::par_unseq, x.begin(), x.end(), x.begin(), [=](auto x) { return x * 2; });
    // reduce
    return std::reduce(std::execution::par_unseq, x.begin(), x.end(), 0);
}

int main(int argc, char *argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);

    // get # of elements that fit in memory
    int numElements = 1 << getSpace(argv[1], sizeof(TYPE), 2);
    printf("Number of elements %d\n", numElements);

    // input vector and tmp vector to verify later
    std::vector<TYPE> x(numElements), tmpX(numElements);

    // init
    srand(time(nullptr));
    for (int i = 0; i < numElements; ++i) {
        x[i] = rand() % 200;
    }
    tmpX = x; // save x as the std::transform is in-place

    auto start = std::chrono::steady_clock::now();
    auto mappedSum = fmr(x);
    auto end = std::chrono::steady_clock::now();

    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    // verify
    auto tmp = 0;
    for (int i = 0; i < numElements; ++i)
        tmp += (tmpX[i] & 1) ? 0 : tmpX[i] * 2;

    if (abs(mappedSum - tmp) > 0) {
        fprintf(stderr, "Result verification failed at element!\n");
        exit(EXIT_FAILURE);
    }
    printf("Test PASSED!\n");
}
