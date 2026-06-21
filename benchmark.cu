#include "utils.h"
#include "benchmark.h"
#include <iostream>

int main() {

    std::vector<int> mSize = {256, 512, 1024, 2048, 4096, 8192};

    for (int item : mSize) {
        std::cout << "Benchmarking for " << item << std::endl;
        int row_A = item;
        int N = item;
        int col_B = item;

        std::vector<float> A_h(row_A * N);
        std::vector<float> B_h(N * col_B);
        std::vector<float> C_h(row_A * col_B, 0.0f);
        float cublasPer = 0.0f;

        printf("Loading matrices from disk...\n");
        loadBinaryWeights("matrix_A_" + std::to_string(item) + ".bin", A_h);
        loadBinaryWeights("matrix_B_" + std::to_string(item) + ".bin", B_h);
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

        printf("\n");
        printf("\n");
        cudaMemset(C_d, 0, bytes_C);
        cublasRef(A_d, B_d, C_d, row_A, N, col_B, cublasPer);
        cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
        verifyCPU(C_h, "matrix_C_ref_" + std::to_string(item) + ".bin", row_A, col_B, N);
        printf("\n");
        printf("\n");

        
        cudaMemset(C_d, 0, bytes_C);
        benchMatrixMulNaive(A_d, B_d, C_d, row_A, N, col_B, cublasPer);
        cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
        verifyCPU(C_h, "matrix_C_ref_" + std::to_string(item) + ".bin", row_A, col_B, N);
        printf("\n");
        printf("\n");

        cudaMemset(C_d, 0, bytes_C);
        benchMatrixMulTiled(A_d, B_d, C_d, row_A, N, col_B, cublasPer);
        cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
        verifyCPU(C_h, "matrix_C_ref_" + std::to_string(item) + ".bin", row_A, col_B, N);
        printf("\n");
        printf("\n");

        cudaMemset(C_d, 0, bytes_C);
        benchMatrixMulCoarsed_1D(A_d, B_d, C_d, row_A, N, col_B, cublasPer);
        cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
        verifyCPU(C_h, "matrix_C_ref_" + std::to_string(item) + ".bin", row_A, col_B, N);
        printf("\n");
        printf("\n");

        cudaMemset(C_d, 0, bytes_C);
        benchMatrixMulCoarsed_2D(A_d, B_d, C_d, row_A, N, col_B, cublasPer);
        cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
        verifyCPU(C_h, "matrix_C_ref_" + std::to_string(item) + ".bin", row_A, col_B, N);
        printf("\n");
        printf("\n");

        cudaMemset(C_d, 0, bytes_C);
        benchMatrixMulWarpTiled(A_d, B_d, C_d, row_A, N, col_B, cublasPer);
        cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
        verifyCPU(C_h, "matrix_C_ref_" + std::to_string(item) + ".bin", row_A, col_B, N);
        printf("\n");
        printf("\n");

        cudaMemset(C_d, 0, bytes_C);
        benchMatrixMulVectorizedLoads(A_d, B_d, C_d, row_A, N, col_B, cublasPer);
        cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);
        verifyCPU(C_h, "matrix_C_ref_" + std::to_string(item) + ".bin", row_A, col_B, N);

        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);

        std::cout << std::endl;
        std::cout << std::endl;
    }
}