#include "cutqc.cu"
#include <iostream>
#include <cuda_runtime.h>


int main() {
    // Do a Kronecker product of |rows| and |2*rows| length vectors
    const int rows = 4;
    const int len_A = rows;
    const int size_A = len_A * sizeof(float);
    const int len_B = rows * 2;
    const int size_B = len_B * sizeof(float);
    const int len_C = rows * rows * 2;
    const int size_C = len_C * sizeof(float);

    float h_A[rows], h_B[rows * 2], h_C[rows * rows * 2];
    for (int i = 0; i < rows; ++i) {
        h_A[i] = i;
        h_B[i] = i;
        h_B[i+rows] = i+rows;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    int total = len_A * len_B;
    int threadsPerBlock = 16;
    int blocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    vectorKron<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, rows*2);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    std::cout << "Input vector A:\n";
    for (int i = 0; i < len_A; ++i) {
        std::cout << h_A[i] << " ";
    }
    std::cout << "\nInput vector B:\n";
    for (int i = 0; i < len_B; ++i) {
        std::cout << h_B[i] << " ";
    }
    std::cout << "\nResult vector C:\n";
    for (int i = 0; i < len_C; ++i) {
        std::cout << h_C[i] << " ";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
