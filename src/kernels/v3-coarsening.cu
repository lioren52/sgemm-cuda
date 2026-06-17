#include "benchmark.h"
#include "utils.h"


#define TILE_WIDTH 32
#define COARSE_FACTOR 4
#define COARSE_FACTOR_2D 2



__global__ void matrixMulCoarse_1D(float* A, float* B, float* C, int row_A, int N, int col_B) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH * COARSE_FACTOR];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float pVal[COARSE_FACTOR] = {0.0f};

    for (int ph = 0; ph < N/TILE_WIDTH; ph++) {
        As[ty][tx] = A[row*N + (ph*TILE_WIDTH+tx)];
        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;
            Bs[ty][tx + (c*TILE_WIDTH)] = B[(ph*TILE_WIDTH + ty)*col_B + col];
        }
        __syncthreads();

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;

            for (int i = 0; i < TILE_WIDTH; i++) {
                pVal[c] += As[ty][i] * Bs[i][tx + (c*TILE_WIDTH)];
            }
        }
        __syncthreads();
    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c*TILE_WIDTH;
        C[row*col_B + col] = pVal[c];
    }
}


void benchMatrixMulCoarsed_1D(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B) {

    dim3 threadPerBlock_coarse(32, 32);
    dim3 blocks_coarse(
        col_B / (threadPerBlock_coarse.x * COARSE_FACTOR),
        row_A / threadPerBlock_coarse.y
    );

    printf("Warming up GPU......\n");
    matrixMulCoarse_1D<<<blocks_coarse, threadPerBlock_coarse>>>(A_d, B_d, C_d, row_A, N, col_B);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("BenchMarking.....Coarsed 1D\n");
    std::vector<float> timings(100);

    for (int i = 0; i < 100; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matrixMulCoarse_1D<<<blocks_coarse, threadPerBlock_coarse>>>(A_d, B_d, C_d, row_A, N, col_B);
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
