#include <vector>
#include <algorithm>
#include <cmath>
#include <execution>
#include <ctime>
#include <chrono>
#include "../helper.h"

#define TYPE float

template<class T>
void vectorAdd(const std::vector<T> &A, const std::vector<T> &B, std::vector<T> &C) {
    std::transform(std::execution::par_unseq, A.begin(), A.end(), B.begin(), C.begin(),
                   [=](auto a, auto b) { return a + b; });
}

void vectorAdd(const int exponent) {
    auto numElements = 1 << exponent;
    printf("Number of elements 2^%d = %d\n", exponent, numElements);

    std::vector<TYPE> A(numElements), B(numElements), C(numElements);

    srand(time(NULL));
    for (int i = 0; i < numElements; ++i) {
        A[i] = rand() / (float) RAND_MAX;
        B[i] = rand() / (float) RAND_MAX;
    }

    auto start = std::chrono::steady_clock::now();
    vectorAdd<TYPE>(A, B, C);
    auto end   = std::chrono::steady_clock::now(); 

    printf("Total time elapsed: %liµs\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    for (int i = 0; i < numElements; ++i) {
        if (fabs(A[i] + B[i] - C[i]) > 1e-5) {
            fprintf(stderr, "Test: FAILED at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test: PASSED!\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) exit(EXIT_FAILURE);
    int maxExponent = getSpace(argv[1], sizeof(float), 3);
    for (int i = 1; i <= maxExponent; ++i) 
        vectorAdd(i);
}
