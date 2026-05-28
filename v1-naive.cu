#include <vector>
#include <iostream>
#include <cstdlib>

__global__ void matrixMul(int* A, int* B, int* C, int row_A, int N, int col_B) {
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    if (Row < row_A && Col < col_B) {
        int pValue = 0;
        for (int i = 0; i < N; i++) {
        pValue += A[(Row*N) + i] * B[Col + (i*col_B)];
        }
        C[(Row*col_B) + Col] = pValue;
    }
}

// CPU Verification adapted for rectangular matrices
void verifyCPU(const std::vector<int>& A, const std::vector<int>& B, const std::vector<int>& C, int row_A, int N, int col_B) {
    printf("Starting CPU verification... (The CPU might take a few seconds to do what the GPU did instantly)\n");
    for (int row = 0; row < row_A; row++) {
        for (int col = 0; col < col_B; col++) {
            int expected = 0;
            for (int i = 0; i < N; i++) {
                expected += A[row * N + i] * B[i * col_B + col];
            }
            if (C[row * col_B + col] != expected) {
                printf("FAIL at [%d, %d]: GPU=%d, CPU=%d\n", row, col, C[row*col_B+col], expected);
                return;
            }
        }
    }
    printf("SUCCESS: GPU Matrix Math matches CPU perfectly across %d elements!\n", row_A * col_B);
}

int main() {
    int row_A = 1024;
    int N = 512;
    int col_B = 1024;

    std::vector<int> A_h(row_A * N);
    std::vector<int> B_h(N * col_B);
    std::vector<int> C_h(row_A * col_B, 0);

    for (int i = 0; i < row_A * N; i++) A_h[i] = rand() % 10;
    for (int i = 0; i < N * col_B; i++) B_h[i] = rand() % 10;

    int* A_d;
    int* B_d;
    int* C_d;

    size_t size_A = row_A * N * sizeof(int);
    size_t size_B = N * col_B * sizeof(int);
    size_t size_C = row_A * col_B * sizeof(int);

    cudaMalloc((void**)&A_d, size_A);
    cudaMalloc((void**)&B_d, size_B);
    cudaMalloc((void**)&C_d, size_C);

    cudaMemcpy(A_d, A_h.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h.data(), size_B, cudaMemcpyHostToDevice);

    dim3 threadPerBlock(16, 16);
    dim3 blocks((col_B + threadPerBlock.x - 1) / threadPerBlock.x, (row_A + threadPerBlock.y - 1) / threadPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMul<<<blocks, threadPerBlock>>>(A_d, B_d, C_d, row_A, N, col_B);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel Execution Time: %f ms\n", milliseconds);
    // --- NEW GFLOPS CALCULATION ---

    // 1. Convert ms to seconds
    double seconds = milliseconds / 1000.0;

    // 2. Calculate total operations (2 * M * N * K)
    double total_ops = 2.0 * (double)row_A * (double)col_B * (double)N;

    // 3. Convert total ops to Giga-ops (divide by 1 billion)
    double giga_ops = total_ops / 1e9;

    // 4. Calculate throughput
    double gflops = giga_ops / seconds;

    printf("Throughput: %f GFLOPs\n", gflops);

    cudaMemcpy(C_h.data(), C_d, size_C, cudaMemcpyDeviceToHost);

    verifyCPU(A_h, B_h, C_h, row_A, N, col_B);


    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);


    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return 0;
}