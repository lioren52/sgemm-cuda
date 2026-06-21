#include "utils.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

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

void loadBinaryWeights(std::string filename, std::vector<float>& vec) {
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
