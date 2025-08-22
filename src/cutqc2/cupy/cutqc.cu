extern "C" __global__
void vectorKron(const float* a, const float* b, float* result, int p, int q) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (q == 1) {
        // Special case: b has size 1
        if (idx < p) {
            result[idx] = a[idx] * b[0];
        }
    } else {
        int total = p * q;
        if (idx < total) {
            int i = idx / q;   // index in a
            int j = idx % q;   // index in b
            result[idx] = a[i] * b[j];
        }
    }
}