#include "utils.h"
#include "benchmark.h"
#include <iostream>

int main() {
    int row_A = 4096;
    int N = 4096;
    int col_B = 4096;

    std::vector<float> A_h(row_A * N);
    std::vector<float> B_h(N * col_B);
    std::vector<float> C_h(row_A * col_B, 0.0f);

    printf("Loading matrices from disk...\n");
    loadBinaryWeights("matrix_A.bin", A_h);
    loadBinaryWeights("matrix_B.bin", B_h);
    printf("Matrices loaded successfully.\n");

    float *A_d, *B_d, *C_d;
    size_t bytes_A = A_h.size() * sizeof(float);
    size_t bytes_B = B_h.size() * sizeof(float);
    size_t bytes_C = C_h.size() * sizeof(float);

    cudaMalloc((void**)&A_d, bytes_A);
    cudaMalloc((void**)&B_d, bytes_B);
    cudaMalloc((void**)&C_d, bytes_C); 

    cudaMemcpy(A_d, A_h.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(), bytes_B, cudaMemcpyHostToDevice);

    benchMatrixMulNaive(A_d, B_d, C_d, row_A, N, col_B);
    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
    verifyCPU(C_h, "matrix_C_ref.bin", row_A, col_B, N);

    benchMatrixMulTiled(A_d, B_d, C_d, row_A, N, col_B);
    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
    verifyCPU(C_h, "matrix_C_ref.bin", row_A, col_B, N);

    benchMatrixMulCoarsed_1D(A_d, B_d, C_d, row_A, N, col_B);
    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
    verifyCPU(C_h, "matrix_C_ref.bin", row_A, col_B, N);

    benchMatrixMulCoarsed_2D(A_d, B_d, C_d, row_A, N, col_B);
    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
    verifyCPU(C_h, "matrix_C_ref.bin", row_A, col_B, N);

    benchMatrixMulWarpTiled(A_d, B_d, C_d, row_A, N, col_B);
    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
    verifyCPU(C_h, "matrix_C_ref.bin", row_A, col_B, N);

    benchMatrixMulVectorizedLoads(A_d, B_d, C_d, row_A, N, col_B);
    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
    verifyCPU(C_h, "matrix_C_ref.bin", row_A, col_B, N);

    cublasRef(A_d, B_d, C_d, row_A, N, col_B);
    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
    verifyCPU(C_h, "matrix_C_ref.bin", row_A, col_B, N);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}