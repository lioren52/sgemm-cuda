#pragma once

void benchMatrixMulNaive(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B);

void benchMatrixMulTiled(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B);

void benchMatrixMulCoarsed_1D(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B);

void benchMatrixMulCaorsed_2D(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B);

void benchMatrixMulWarpTiled(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B);

void benchMatrixMulVectorizedLoads(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B);

void cublasRef(float *A_d, float *B_d, float *C_d, int row_A, int N, int col_B);
