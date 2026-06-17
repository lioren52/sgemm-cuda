#include "benchmark.h"
#include "utils.h"


#define BM 64
#define BN 64
#define BK 32
#define WM 16
#define WN 32
#define TM 4
#define TN 4

__global__ void matrixMulVectorizedLoads(float* A, float* B, float* C, int M, int K, int N) {
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

        for (int i = 0; i < (BM * BK) / (total_threads * 4); i++) {
            int idx  = tid + i * total_threads;
            int row  = idx / (BK / 4);      // which row of As
            int col4 = idx % (BK / 4);      // which float4 chunk in that row (0..7)

            float4 tmp = reinterpret_cast<float4*>(
                &A[(blockIdx.y * BM + row) * K + ph * BK + col4 * 4]
            )[0];

            As[row][col4 * 4 + 0] = tmp.x;
            As[row][col4 * 4 + 1] = tmp.y;
            As[row][col4 * 4 + 2] = tmp.z;
            As[row][col4 * 4 + 3] = tmp.w;
        }

        for (int i = 0; i < (BK * BN) / (total_threads * 4); i++) {
            int idx  = tid + i * total_threads;
            int row  = idx / (BN / 4);
            int col4 = idx % (BN / 4);

            float4 tmp = reinterpret_cast<float4*>(
                &B[(ph * BK + row) * N + blockIdx.x * BN + col4 * 4]
            )[0];

            Bs[row][col4 * 4 + 0] = tmp.x;
            Bs[row][col4 * 4 + 1] = tmp.y;
            Bs[row][col4 * 4 + 2] = tmp.z;
            Bs[row][col4 * 4 + 3] = tmp.w;
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


void benchMatrixMulVectorizedLoads(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B) {

    dim3 threadPerBlock_Vector(16, 16);
    dim3 blocks_Vector(
        (col_B + BN - 1) / BN,
        (row_A + BM - 1) / BM
    );

    printf("Warming up GPU......\n");
    matrixMulVectorizedLoads<<<blocks_Vector, threadPerBlock_Vector>>>(A_d, B_d, C_d, row_A, N, col_B);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("BenchMarking.....Vectorized Loading\n");
    std::vector<float> timings(100);

    for (int i = 0; i < 100; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matrixMulVectorizedLoads<<<blocks_Vector, threadPerBlock_Vector>>>(A_d, B_d, C_d, row_A, N, col_B);
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
