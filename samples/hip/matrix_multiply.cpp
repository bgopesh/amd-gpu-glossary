// Simple HIP Matrix Multiplication
// Usage: ./matrix_multiply [size]

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

#define TILE_SIZE 16

__global__ void matrixMul(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    // Matrix size (N x N)
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t bytes = N * N * sizeof(float);

    std::cout << "Matrix size: " << N << " x " << N << std::endl;

    // Allocate host memory
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(matrixMul, gridSize, blockSize, 0, 0, d_A, d_B, d_C, N);
    HIP_CHECK(hipGetLastError());

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));

    // Verify result (spot check)
    bool success = true;
    float expected = N * 2.0f;  // Each element should be N * 1.0 * 2.0
    for (int i = 0; i < 10; i++) {
        if (fabs(h_C[i] - expected) > 0.01f) {
            success = false;
            break;
        }
    }

    std::cout << "Result: " << (success ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return 0;
}
