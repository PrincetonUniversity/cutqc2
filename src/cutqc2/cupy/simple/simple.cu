extern "C" __global__ void matrixAdd(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = row * cols + col;
    if (row < rows && col < cols) {
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" __global__ void matrixSubtract(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = row * cols + col;
    if (row < rows && col < cols) {
        C[idx] = A[idx] - B[idx];
    }
}

extern "C" __global__
void vectorKron(const float* a, const float* b, float* result, int n, int m) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = n * m;
    if (idx < total) {
        int i = idx / m;   // index in a
        int j = idx % m;   // index in b
        result[idx] = a[i] * b[j];
    }
}