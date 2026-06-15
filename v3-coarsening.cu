#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4
#define COARSE_FACTOR_2D 2


__global__ void matrixMulCoarse(float* A, float* B, float* C, int row_A, int N, int col_B) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float pVal[COARSE_FACTOR] = {0.0f};

    for (int ph = 0; ph < N/TILE_WIDTH; ph++) {
        As[ty][tx] = A[row*N + (ph*TILE_WIDTH+tx)];

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = colStart + c * TILE_WIDTH;

            Bs[ty][tx] = B[(ph*TILE_WIDTH + ty)*col_B + col];
            __syncthreads();

            for (int i = 0; i < TILE_WIDTH; i++) {
                pVal[c] += As[ty][i] * Bs[i][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        int col = colStart + c*TILE_WIDTH;
        C[row*col_B + col] = pVal[c];
    }
}

__global__ void matrixMulCoarse_v2(float* A, float* B, float* C, int row_A, int N, int col_B) {
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

void benchMatrixMulCoarsed() {
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

    dim3 threadPerBlock_coarse(32, 32);
    dim3 blocks_coarse(
        col_B / (threadPerBlock_coarse.x * COARSE_FACTOR),
        row_A / threadPerBlock_coarse.y
    );

    dim3 threadPerBlock_2D(32, 32);
    dim3 blocks_2D(
        col_B / (TILE_WIDTH * COARSE_FACTOR_2D),
        row_A / (TILE_WIDTH * COARSE_FACTOR_2D)
    );

    printf("Warming up GPU......\n");
    matrixMulCoarse<<<blocks_coarse, threadPerBlock>>>(A_d, B_d, C_d, row_A, N, col_B);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("BenchMarking.....Coarsed\n");
    std::vector<float> timings(100);

    for (int i = 0; i < 100; i++) {
        cudaMemset(C_d, 0, bytes_C);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matrixMulCoarse<<<blocks_coarse, threadPerBlock_coarse>>>(A_d, B_d, C_d, row_A, N, col_B);
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

    printf("\n");
    printf("\n");
    printf("\n");

    
    printf("BenchMarking.....Coarsed v2\n");

    for (int i = 0; i < 100; i++) {
        cudaMemset(C_d, 0, bytes_C);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        matrixMulCoarse_v2<<<blocks_coarse, threadPerBlock_coarse>>>(A_d, B_d, C_d, row_A, N, col_B);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&timings[i], start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::sort(timings.begin(), timings.end());
    median_ms = timings[50];
    printf("Timings\n");
    printf("Median kernel time: %f ms\n", median_ms);
    printf("Min kernel time: %f ms\n", timings[0]);
    printf("Max kernel time: %f ms\n", timings[99]);

    printf("\n");
    printf("GFLOPs\n");
    medianG = time2GFLOPs(timings[50], row_A, col_B, N);
    minG = time2GFLOPs(timings[99], row_A, col_B, N);
    maxG = time2GFLOPs(timings[0], row_A, col_B, N);
    printf("Median GFLOPs: %f\n", medianG);
    printf("Min GFLOPs: %f\n", minG);
    printf("Max GFLOPs: %f\n", maxG);

    printf("\n");
    printf("Clock Stats\n");
    clock_mhz = prop.clockRate / 1000;
    sm_count = prop.multiProcessorCount;
    peak_gflops = 2.0 * sm_count * 64 * clock_mhz * 1e6 / 1e9;

    printf("Current clock: %d MHz\n", clock_mhz);
    printf("Theoretical peak at this clock: %.1f GFLOPs\n", peak_gflops);
    printf("Kernel efficiency: %.1f%%\n", medianG / peak_gflops * 100.0);

    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);

    verifyCPU(C_h, "matrix_C_ref.bin", row_A, col_B, N);

    printf("\n");
    printf("\n");
    printf("\n");

    printf("BenchMarking.....Coarsed 2D\n");

    for (int i = 0; i < 100; i++) {
        cudaMemset(C_d, 0, bytes_C);
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
    median_ms = timings[50];

    printf("Timings\n");
    printf("Median kernel time: %f ms\n", median_ms);
    printf("Min kernel time: %f ms\n", timings[0]);
    printf("Max kernel time: %f ms\n", timings[99]);

    printf("\n");
    printf("GFLOPs\n");
    medianG = time2GFLOPs(timings[50], row_A, col_B, N);
    minG = time2GFLOPs(timings[99], row_A, col_B, N);
    maxG = time2GFLOPs(timings[0], row_A, col_B, N);
    printf("Median GFLOPs: %f\n", medianG);
    printf("Min GFLOPs: %f\n", minG);
    printf("Max GFLOPs: %f\n", maxG);

    printf("\n");
    printf("Clock Stats\n");

    clock_mhz = prop.clockRate / 1000;
    sm_count = prop.multiProcessorCount;
    peak_gflops = 2.0 * sm_count * 64 * clock_mhz * 1e6 / 1e9;

    printf("Current clock: %d MHz\n", clock_mhz);
    printf("Theoretical peak at this clock: %.1f GFLOPs\n", peak_gflops);
    printf("Kernel efficiency: %.1f%%\n", medianG / peak_gflops * 100.0);

    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);

    verifyCPU(C_h, "matrix_C_ref.bin", row_A, col_B, N);
}
