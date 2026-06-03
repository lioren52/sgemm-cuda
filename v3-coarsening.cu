#include <iostream>
#include <vector>

#define TILE_WIDTH 32
#define COARSE_FACTOR 4


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

void verifyCPU(const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C, int row_A, int N, int col_B) {
    printf("Starting CPU verification... (The CPU might take a few seconds)\n");
  
    float epsilon = 0.1f; 

    for (int row = 0; row < row_A; row++) {
        for (int col = 0; col < col_B; col++) {
            float expected = 0.0f;
            for (int i = 0; i < N; i++) {
                expected += A[row * N + i] * B[i * col_B + col];
            }
            
            float actual = C[row * col_B + col];
            
            if (std::abs(actual - expected) > epsilon) {
                printf("FAIL at [%d, %d]: GPU=%f, CPU=%f\n", row, col, actual, expected);
                return;
            }
        }
    }
    printf("SUCCESS: GPU Matrix Math matches CPU perfectly across %d elements!\n", row_A * col_B);
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

int main() {
    int row_A = 1024;
    int N = 512;
    int col_B = 1024;

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

    dim3 threadPerBlock(32, 32);
    dim3 blocks(col_B / (threadPerBlock.x * COARSE_FACTOR), row_A / threadPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMulCoarse<<<blocks, threadPerBlock>>>(A_d, B_d, C_d, row_A, N, col_B);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FATAL CUDA ERROR: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel Execution Time: %f ms\n", milliseconds);

    double seconds = milliseconds / 1000.0;
    double total_ops = 2.0 * (double)row_A * (double)col_B * (double)N;
    double giga_ops = total_ops / 1e9;
    double gflops = giga_ops / seconds;

    printf("Throughput: %f GFLOPs\n", gflops);

    cudaMemcpy(C_h.data(), C_d, bytes_C, cudaMemcpyDeviceToHost);

    verifyCPU(A_h, B_h, C_h, row_A, N, col_B);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}