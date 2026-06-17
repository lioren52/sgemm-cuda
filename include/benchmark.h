#pragma once

void benchMatrixMulNaive(float *A_d, float *B_d, float *C_d, int& row_A, int& N, int& col_B, float& cublasPer);

void benchMatrixMulTiled(float *A_d, float *B_d, float *C_d, int& row_A, int& N, int& col_B, float& cublasPer);

void benchMatrixMulCoarsed_1D(float *A_d, float *B_d, float *C_d, int& row_A, int& N, int& col_B, float& cublasPer);

void benchMatrixMulCoarsed_2D(float *A_d, float *B_d, float *C_d, int& row_A, int& N, int& col_B, float& cublasPer);

void benchMatrixMulWarpTiled(float *A_d, float *B_d, float *C_d, int& row_A, int& N, int& col_B, float& cublasPer);

void benchMatrixMulVectorizedLoads(float *A_d, float *B_d, float *C_d, int& row_A, int& N, int& col_B, float& cublasPer);

void cublasRef(float *A_d, float *B_d, float *C_d, int& row_A, int& N, int& col_B, float& cublasPer);
