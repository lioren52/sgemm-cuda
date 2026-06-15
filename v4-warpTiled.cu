#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

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

void verifyCPU(const std::vector<float>& C_gpu, const char* ref_file, int row_A, int col_B, int N) {
    printf("Starting verification against reference...\n");

    std::ifstream in(ref_file, std::ios::binary);
    if (!in) {
        printf("ERROR: %s not found. Run generator first.\n", ref_file);
        return;
    }

    std::vector<float> C_ref(row_A * col_B);
    in.read(reinterpret_cast<char*>(C_ref.data()), C_ref.size() * sizeof(float));
    in.close();

    float epsilon = 1e-2f * N;
    for (int row = 0; row < row_A; row++) {
        for (int col = 0; col < col_B; col++) {
            int idx = row * col_B + col;
            if (std::abs(C_gpu[idx] - C_ref[idx]) > epsilon) {
                printf("FAIL at [%d, %d]: GPU=%f, REF=%f\n", row, col, C_gpu[idx], C_ref[idx]);
                return;
            }
        }
    }
    printf("SUCCESS: %d elements verified against reference.\n", row_A * col_B);
}

void loadBinaryWeights(const char* filename, std::vector<float>& vec) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        printf("FATAL ERROR: Could not find %s. Did you run the generator?\n", filename);
        exit(1);
    }
    
    // Read the exact byte footprint straight into the vector's memory
    in.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(float));
    in.close();
}

double time2GFLOPs(float milliseconds, int row_A, int col_B, int N) {
    double seconds = milliseconds / 1000.0;
    double total_ops = 2.0 * (double)row_A * (double)col_B * (double)N;
    double giga_ops = total_ops / 1e9;
    double gflops = giga_ops / seconds;
    return gflops;
}

void benchMatrixMulWarpTiled() {
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
        cudaMemset(C_d, 0, bytes_C);
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
}