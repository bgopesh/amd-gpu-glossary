// Simple HIP Vector Addition
// Usage: ./vector_add [size]

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

#define HIP_CHECK(cmd) \
{ \
    hipError_t error = cmd; \
    if (error != hipSuccess) { \
        std::cerr << "Error: '" << hipGetErrorString(error) << "' (" << error << ") at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char** argv) {
    // Vector size
    int n = (argc > 1) ? atoi(argv[1]) : 1000000;
    size_t bytes = n * sizeof(float);

    std::cout << "Vector size: " << n << std::endl;

    // Allocate host memory
    float* h_a = new float[n];
    float* h_b = new float[n];
    float* h_c = new float[n];

    // Initialize vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(vectorAdd, dim3(gridSize), dim3(blockSize), 0, 0, d_a, d_b, d_c, n);
    HIP_CHECK(hipGetLastError());

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));

    // Verify result
    bool success = true;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            break;
        }
    }

    std::cout << "Result: " << (success ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    return 0;
}
