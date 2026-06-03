#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

void generateAndSaveBinary(const char* filename, size_t elements) {
    std::vector<float> vec(elements);
    
    // Fill with your standard logic
    for (size_t i = 0; i < elements; i++) {
        vec[i] = (float)(rand() % 10);
    }
    
    // Open file in raw binary mode
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "FATAL ERROR: Could not open " << filename << " for writing.\n";
        exit(1);
    }
    
    // Blast the raw memory straight to disk
    out.write(reinterpret_cast<const char*>(vec.data()), elements * sizeof(float));
    out.close();
    
    std::cout << "SUCCESS: Wrote " << (elements * sizeof(float)) / (1024 * 1024.0) 
              << " MB to " << filename << "\n";
}

int main() {
    size_t row_A = 4096;
    size_t N = 4096;
    size_t col_B = 4096;

    std::cout << "Generating Matrix A...\n";
    generateAndSaveBinary("matrix_A.bin", row_A * N);
    
    std::cout << "Generating Matrix B...\n";
    generateAndSaveBinary("matrix_B.bin", N * col_B);

    return 0;
}