#include "benchmark.h"
#include "utils.h"

#define TILE_WIDTH 32
#define COARSE_FACTOR 4
#define COARSE_FACTOR_2D 2

__global__ void matrixMulCoarse_2D(float* A, float* B, float* C, int row_A, int N, int col_B) {
    __shared__ float As[TILE_WIDTH * COARSE_FACTOR_2D][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH * COARSE_FACTOR_2D];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int rowStart = by * TILE_WIDTH * COARSE_FACTOR_2D + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR_2D + tx;

    float pVal[COARSE_FACTOR_2D][COARSE_FACTOR_2D] = {0.0f};

    for (int ph = 0; ph < N/TILE_WIDTH; ph++) {
        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            int col = colStart + c * TILE_WIDTH;
            int row = rowStart + c * TILE_WIDTH;
            Bs[ty][tx + (c*TILE_WIDTH)] = B[(ph*TILE_WIDTH + ty)*col_B + col];
            As[ty + (c*TILE_WIDTH)][tx] = A[row*N + (ph*TILE_WIDTH+tx)];
        }
        __syncthreads();

        for (int c = 0; c < COARSE_FACTOR_2D; c++) {
            for (int r = 0; r < COARSE_FACTOR_2D; r++) {
                for (int i = 0; i < TILE_WIDTH; i++) {
                    pVal[r][c] += As[ty + (r*TILE_WIDTH)][i] * Bs[i][tx + (c*TILE_WIDTH)];
                }
            }
        }
        __syncthreads();
    }

    for (int c = 0; c < COARSE_FACTOR_2D; c++) {
        for (int r = 0; r < COARSE_FACTOR_2D; r++) {
            int col = colStart + c * TILE_WIDTH;
            int row = rowStart + r * TILE_WIDTH;
            C[row*col_B + col] = pVal[r][c];
        }
    }
}

void benchMatrixMulCoarsed_2D(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B) {

    dim3 threadPerBlock_2D(32, 32);
    dim3 blocks_2D(
        col_B / (TILE_WIDTH * COARSE_FACTOR_2D),
        row_A / (TILE_WIDTH * COARSE_FACTOR_2D)
    );

    printf("Warming up GPU......\n");
    matrixMulCoarse_2D<<<blocks_2D, threadPerBlock_2D>>>(A_d, B_d, C_d, row_A, N, col_B);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("BenchMarking.....Coarsed\n");
    std::vector<float> timings(100);

    for (int i = 0; i < 100; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matrixMulCoarse_2D<<<blocks_2D, threadPerBlock_2D>>>(A_d, B_d, C_d, row_A, N, col_B);
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

    int clock_mhz = prop.clockRate / 1000;
    int sm_count = prop.multiProcessorCount;
    double peak_gflops = 2.0 * sm_count * 64 * clock_mhz * 1e6 / 1e9;

    printf("\n");
    printf("Clock Stats\n");
    printf("Current clock: %d MHz\n", clock_mhz);
    printf("Theoretical peak at this clock: %.1f GFLOPs\n", peak_gflops);
    printf("Kernel efficiency: %.1f%%\n", medianG / peak_gflops * 100.0);
}