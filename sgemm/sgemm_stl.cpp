#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <cstdio>
#include <chrono>

#define TYPE double
#define N 1024

template<class T>
using Matrix = std::vector<std::vector<T>>;

template<class T>
Matrix<T> sgemm(Matrix<T> const &A, Matrix<T> const &B) {
    auto rows = A.size();
    auto cols = B[0].size();
    Matrix<T> result;

    // transpose matrix B
    Matrix<T> tB(B[0].size(), std::vector<T>(B.size()));
#pragma omp parallel for collapse(2)
    for (auto r = 0; r < B.size(); ++r)
            for (auto c = 0; c < B[0].size(); ++c)
                tB[c][r] = B[r][c];

    // parallel inner_product
#pragma unroll
    for (auto &a: A) {
        std::vector<T> tmp(cols);
#pragma omp parallel for
        for (auto i = 0; i < tB.size(); ++i)
            tmp[i] = std::transform_reduce(std::execution::par_unseq, a.begin(), a.end(), tB[i].begin(), 0.0);
        result.push_back(std::move(tmp));
    }

    return result;
}

int main() {
    /*
    Matrix<TYPE> A = {
            {2, 3},
            {1, 2},
            {1, 1}
    };
    Matrix<TYPE> B = {
            {3, 1, 2},
            {4, 1, 1}
    };
     */
    Matrix<TYPE> A(N, std::vector<TYPE>(N));
    Matrix<TYPE> B(N, std::vector<TYPE>(N));
    std::srand(time(nullptr));
    for (auto n = 0; n < N; n++)
        for (auto m = 0; m < N; m++)
            A[n][m] = rand() / (double) RAND_MAX;
    for (auto o = 0; o < N; o++)
        for (auto p = 0; p < N; p++)
            B[o][p] = rand() / (double) RAND_MAX;

    auto start = std::chrono::steady_clock::now();
    Matrix<TYPE> C = sgemm<TYPE>(A, B);
    auto end = std::chrono::steady_clock::now();

    printf("Total time elapsed: %lims\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    printf("result matrix has %zu rows and %zu cols\n", C.size(), C[0].size());

    /*
    for (auto &c: C) {
        for (auto &cell: c)
            printf("%f\t", cell);
        printf("\n");
    }
    printf("\n");
    */
}
