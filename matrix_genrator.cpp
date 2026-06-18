#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

void saveBinary(const char* filename, const std::vector<float>& vec) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "FATAL ERROR: Could not open " << filename << "\n";
        exit(1);
    }
    out.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
    out.close();
    std::cout << "Wrote " << (vec.size() * sizeof(float)) / (1024 * 1024.0)
              << " MB to " << filename << "\n";
}

int main() {
    size_t row_A = 8192;
    size_t N     = 8192;
    size_t col_B = 8192;

    srand(42);

    std::cout << "Generating Matrix A...\n";
    std::vector<float> A(row_A * N);
    for (size_t i = 0; i < A.size(); i++)
        A[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    saveBinary("matrix_A.bin", A);

    std::cout << "Generating Matrix B...\n";
    std::vector<float> B(N * col_B);
    for (size_t i = 0; i < B.size(); i++)
        B[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    saveBinary("matrix_B.bin", B);

    std::cout << "Computing reference C = A x B on CPU...\n";
    std::vector<float> C(row_A * col_B, 0.0f);
    for (size_t row = 0; row < row_A; row++) {
        if (row % 256 == 0)
            printf("  Progress: %zu / %zu rows\n", row, row_A);
        for (size_t col = 0; col < col_B; col++) {
            float sum = 0.0f;
            for (size_t i = 0; i < N; i++)
                sum += A[row * N + i] * B[i * col_B + col];
            C[row * col_B + col] = sum;
        }
    }
    saveBinary("matrix_C_ref.bin", C);
    std::cout << "Done. Reference result saved to matrix_C_ref.bin\n";

    return 0;
}