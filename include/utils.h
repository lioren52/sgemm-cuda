#pragma once

#include <vector>
#include <iostream>

void verifyCPU(const std::vector<float>& C_gpu, const char* ref_file, int row_A, int col_B, int N);

void loadBinaryWeights(const char* filename, std::vector<float>& vec);

double time2GFLOPs(float milliseconds, int row_A, int col_B, int N);