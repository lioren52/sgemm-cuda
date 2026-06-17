#include <cublas_v2.h>
#include "benchmark.h"
#include "utils.h"



void cublasRef(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B) {

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta  = 0.0f;

    printf("Warming up GPU......\n");
    cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    col_B, row_A, N,
                    &alpha,
                    B_d, col_B,
                    A_d, N,
                    &beta,
                    C_d, col_B);
    cudaDeviceSynchronize();

    printf("BenchMarking.....cuBLAS\n");
    std::vector<float> timings(100);

    for (int i = 0; i < 100; i++) {
        cudaMemset(C_d, 0, bytes_C);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    col_B, row_A, N,
                    &alpha,
                    B_d, col_B,
                    A_d, N,
                    &beta,
                    C_d, col_B);
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

    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);

    verifyCPU(C_h, "matrix_C_ref.bin", row_A, col_B, N);

    cublasDestroy(handle);
}