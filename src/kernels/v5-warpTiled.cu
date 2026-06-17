#include "benchmark.h"
#include "utils.h"


#define BM 64
#define BN 64
#define BK 32
#define WM 16
#define WN 32
#define TM 4
#define TN 4

__global__ void matrixMulWarpTiled(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid     = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int warp_row = warp_id / 2;
    int warp_col = warp_id % 2;

    int lane_row = lane_id / 8;
    int lane_col = lane_id % 8;

    int out_row = blockIdx.y * BM + warp_row * WM + lane_row * TM;
    int out_col = blockIdx.x * BN + warp_col * WN + lane_col * TN;

    float reg_C[TM][TN] = {0.0f};
    float reg_A[TM];
    float reg_B[TN];

    int total_threads = blockDim.x * blockDim.y;

    for (int ph = 0; ph < K / BK; ph++) {

        for (int i = 0; i < BM * BK / total_threads; i++) {
            int idx = tid + i * total_threads;
            int row = idx / BK;
            int col = idx % BK;
            As[row][col] = A[(blockIdx.y * BM + row) * K + ph * BK + col];
        }

        for (int i = 0; i < BK * BN / total_threads; i++) {
            int idx = tid + i * total_threads;
            int row = idx / BN;
            int col = idx % BN;
            Bs[row][col] = B[(ph * BK + row) * N + blockIdx.x * BN + col];
        }

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            for (int m = 0; m < TM; m++)
                reg_A[m] = As[warp_row * WM + lane_row * TM + m][k];

            for (int n = 0; n < TN; n++)
                reg_B[n] = Bs[k][warp_col * WN + lane_col * TN + n];

            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    reg_C[m][n] += reg_A[m] * reg_B[n];
        }

        __syncthreads();
    }

    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            if (out_row + m < M && out_col + n < N)
                C[(out_row + m) * N + out_col + n] = reg_C[m][n];
}


void benchMatrixMulWarpTiled(float *A_d, float *B_d, float *C_d, int& row_A, int& N, int& col_B, float& cublasPer) {

    dim3 threadPerBlock_Warp(16, 16);
    dim3 blocks_Warp(
        (col_B + BN - 1) / BN,
        (row_A + BM - 1) / BM
    );

    printf("Warming up GPU......\n");
    matrixMulWarpTiled<<<blocks_Warp, threadPerBlock_Warp>>>(A_d, B_d, C_d, row_A, N, col_B);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("BenchMarking.....Warp Tiled\n");
    std::vector<float> timings(100);

    for (int i = 0; i < 100; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matrixMulWarpTiled<<<blocks_Warp, threadPerBlock_Warp>>>(A_d, B_d, C_d, row_A, N, col_B);
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
