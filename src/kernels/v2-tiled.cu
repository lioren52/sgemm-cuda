#include "benchmark.h"
#include "utils.h"


#define TILE_WIDTH_T 16


__global__ void matrixMulTiled(float* A, float* B, float* C, int row_A, int N, int col_B) {
    __shared__ float As[TILE_WIDTH_T][TILE_WIDTH_T];
    __shared__ float Bs[TILE_WIDTH_T][TILE_WIDTH_T];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int col = bx * blockDim.x + tx;
    int row = by * blockDim.y + ty;

    float pVal = 0.0f; 

    for (int ph = 0; ph < (N + TILE_WIDTH_T - 1) / TILE_WIDTH_T; ph++) {
        if (row < row_A && (ph * TILE_WIDTH_T + tx) < N) {
            As[ty][tx] = A[row * N + ph * TILE_WIDTH_T + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((ph * TILE_WIDTH_T + ty) < N && col < col_B) {
            Bs[ty][tx] = B[(ph * TILE_WIDTH_T + ty) * col_B + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH_T; k++) {
            pVal += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < row_A && col < col_B) {
        C[row * col_B + col] = pVal;
    } 
}

void benchMatrixMulTiled(float *A_d, float *B_d, float *C_d, int& row_A, int& N, int& col_B, float& cublasPer) {

    dim3 threadPerBlock(16, 16);
    dim3 blocks((col_B + threadPerBlock.x - 1) / threadPerBlock.x, (row_A + threadPerBlock.y - 1) / threadPerBlock.y);

    printf("Warming up GPU......\n");
    matrixMulTiled<<<blocks, threadPerBlock>>>(A_d, B_d, C_d, row_A, N, col_B);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("BenchMarking.....Tiled\n");
    std::vector<float> timings(100);

    for (int i = 0; i < 100; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matrixMulTiled<<<blocks, threadPerBlock>>>(A_d, B_d, C_d, row_A, N, col_B);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&timings[i], start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::sort(timings.begin(), timings.end());
    float median_ms = timings[50];
    printf("Timings\n");
    printf("Median kernel time: %f ms\n", median_ms);
    printf("Min kernel time: %f ms\n", timings[0]);
    printf("Max kernel time: %f ms\n", timings[99]);
    printf("\n");
    printf("GFLOPs\n");
    double medianG = time2GFLOPs(timings[50], row_A, col_B, N);
    double minG = time2GFLOPs(timings[99], row_A, col_B, N);
    double maxG = time2GFLOPs(timings[0], row_A, col_B, N);
    printf("Median GFLOPs: %f\n", medianG);
    printf("Min GFLOPs: %f\n", minG);
    printf("Max GFLOPs: %f\n", maxG);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    float relPerformance = (medianG/cublasPer) * 100;
    std::cout << "Relative to cuBLAS: " << relPerformance << std::endl;

}
