#include <vector>
#include <algorithm>
#include <random>
#include <execution>
#include <ctime>
#include <chrono>
#include "../helper.h"

#define TYPE int

template <class T>
T reduce(const std::vector<T> v) {
    return std::reduce(std::execution::par_unseq, v.begin(), v.end(), 0.0f);
}

int main(int argc, char* argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);
    int numElements = 1 << getSpace(argv[1], sizeof(TYPE), 2);
    printf("Number of elements %d\n", numElements);

    std::vector<TYPE> x(numElements);

    srand(time(NULL));
    for (int i = 0; i < numElements; ++i)
        x[i] = rand() / (float) RAND_MAX;

    auto start = std::chrono::steady_clock::now();
    auto sum = reduce<TYPE>(x);
    auto end = std::chrono::steady_clock::now();

    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    auto tmp = x[0];
    for (auto i = 1; i < x.size(); ++i) { tmp += x[i]; }
    if (fabs(sum - tmp) > 0.5) {
        fprintf(stderr, "Result verification failed!\n");
        exit(EXIT_FAILURE);
    }
    printf("Test PASSED!\n");
}
