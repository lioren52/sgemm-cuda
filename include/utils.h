#pragma once

#include <vector>
#include <iostream>
#include <algorithm>

void verifyCPU(const std::vector<float>& C_gpu, std::string ref_file, int row_A, int col_B, int N);

void loadBinaryWeights(std::string filename, std::vector<float>& vec);

double time2GFLOPs(float milliseconds, int row_A, int col_B, int N);