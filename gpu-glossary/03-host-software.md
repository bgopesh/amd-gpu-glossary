# Host Software

Development tools, APIs, libraries, and platform software for AMD GPU programming.

## ROCm (Radeon Open Compute platform)

AMD's open-source platform for GPU computing, providing the complete software stack for AMD Instinct GPUs.

**Key components:**
- HIP programming language and runtime
- ROCm runtime and drivers
- Math libraries (rocBLAS, rocFFT, rocRAND, etc.)
- ML frameworks (PyTorch, TensorFlow support)
- Profiling and debugging tools

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install rocm-hip-sdk
```

**Related:** [HIP](#hip-heterogeneous-compute-interface-for-portability), [rocm-smi](#rocm-smi)

## hipcc

The HIP compiler driver that compiles HIP code for AMD GPUs (and optionally NVIDIA GPUs).

**Usage:**
```bash
# Compile for AMD GPU
hipcc mykernel.cpp -o myprogram

# Specify GPU architecture
hipcc --offload-arch=gfx90a mykernel.cpp -o myprogram
```

**Key features:**
- Based on Clang/LLVM
- Supports C++17 and modern C++ features
- Can target multiple GPU architectures
- Integrates with standard build systems

**Related:** [HIP](#hip-heterogeneous-compute-interface-for-portability), [AMDGPU LLVM](#amdgpu-llvm)

## rocm-smi

The ROCm System Management Interface - a command-line tool for monitoring and managing AMD GPUs.

**Common commands:**
```bash
# Show all GPUs
rocm-smi

# Detailed info for specific GPU
rocm-smi -d 0

# Monitor GPU utilization
rocm-smi --showuse

# Show memory usage
rocm-smi --showmeminfo

# Show temperature and power
rocm-smi --showtemp --showpower

# Reset GPU
rocm-smi --gpureset
```

**Related:** [ROCm](#rocm-radeon-open-compute-platform), [Monitoring](#gpu-monitoring)

## rocBLAS

AMD's optimized BLAS (Basic Linear Algebra Subprograms) library for GPU acceleration.

**Key features:**
- Optimized for AMD GPUs
- Full BLAS Level 1, 2, and 3 operations
- Leverages Matrix Core Engines on CDNA GPUs
- Drop-in replacement for cuBLAS

**Example operations:**
- GEMM (General Matrix Multiply)
- GEMV (Matrix-Vector multiplication)
- Dot products, norms, scaling

**Related:** [Matrix Core Engine](#matrix-core-engine), [rocSOLVER](#rocsolver)

## rocFFT

AMD's Fast Fourier Transform library optimized for GPUs.

**Key features:**
- 1D, 2D, and 3D transforms
- Real and complex transforms
- In-place and out-of-place operations
- Optimized for CDNA/RDNA architectures

**Related:** [rocBLAS](#rocblas), [ROCm](#rocm-radeon-open-compute-platform)

## rocRAND

GPU-accelerated random number generation library.

**Supported generators:**
- Pseudo-random: XORWOW, MRG32k3a, MTGP32, Philox
- Quasi-random: Sobol
- Distributions: Uniform, Normal, Log-normal, Poisson

**Related:** [ROCm](#rocm-radeon-open-compute-platform)

## rocSOLVER

Linear algebra library providing LAPACK-like functionality on GPUs.

**Key features:**
- LU, QR, Cholesky decompositions
- Eigenvalue and singular value solvers
- Built on top of rocBLAS
- Optimized for AMD GPUs

**Related:** [rocBLAS](#rocblas), [Matrix Operations](#matrix-operations)

## rocSPARSE

Sparse linear algebra library for GPU acceleration.

**Key operations:**
- Sparse matrix-vector multiplication (SpMV)
- Sparse matrix-matrix multiplication (SpMM)
- Triangular solvers
- Format conversions (CSR, COO, ELL, etc.)

**Related:** [rocBLAS](#rocblas), [Sparse Computing](#sparse-computing)

## MIOpen

AMD's open-source library for high-performance deep learning primitives.

**Key features:**
- Convolution operations (forward, backward)
- Pooling, normalization, activation functions
- RNN and LSTM operations
- Auto-tuning for optimal kernel selection
- Equivalent to NVIDIA's cuDNN

**Related:** [ROCm](#rocm-radeon-open-compute-platform), [Deep Learning](#deep-learning-frameworks)

## RCCL (ROCm Communication Collectives Library)

Multi-GPU communication library for collective operations in distributed training.

**Supported operations:**
- AllReduce, Broadcast, Reduce
- AllGather, ReduceScatter
- Send/Receive (point-to-point)

**Key features:**
- Optimized for AMD Infinity Fabric
- Multi-node support
- Equivalent to NVIDIA's NCCL

**Related:** [Infinity Fabric](#infinity-fabric), [Multi-GPU](#multi-gpu-training)

## PyTorch with ROCm

AMD's officially supported PyTorch distribution with ROCm backend.

**Installation:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

**Key features:**
- Native HIP backend (no CUDA needed)
- Support for MI300, MI250, MI210, MI100
- Mixed precision training (FP16, BF16)
- Distributed training with RCCL

**Related:** [ROCm](#rocm-radeon-open-compute-platform), [MIOpen](#miopen), [RCCL](#rccl-rocm-communication-collectives-library)

## TensorFlow with ROCm

AMD-optimized TensorFlow for GPU acceleration.

**Installation:**
```bash
docker pull rocm/tensorflow:latest
```

**Key features:**
- Official AMD support for ROCm
- XLA compilation support
- Distributed training capabilities
- Works with MI series GPUs

**Related:** [ROCm](#rocm-radeon-open-compute-platform), [MIOpen](#miopen)

## rocprofiler-sdk

Low-level profiling tool for AMD GPUs providing detailed performance metrics. Successor to rocProfiler.

**Key features:**
- Hardware performance counters
- Kernel execution time
- Memory bandwidth utilization
- Wavefront occupancy
- API trace (HIP, HSA)
- Programmatic profiling API
- Kernel dispatch correlation

**Usage:**
```bash
# Collect hardware performance counters
rocprofv3 --pmc --counter SQ_WAVES,SQ_WAVE_CYCLES ./myprogram

# API tracing
rocprofv3 --hip-trace ./myprogram
rocprofv3 --hsa-trace ./myprogram

# Kernel statistics
rocprofv3 --stats ./myprogram
```

**Repository:** [rocprofiler-sdk](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk)

**Related:** [rocprofiler-compute](#rocprofiler-compute-omniperf), [Performance Analysis](#performance-analysis)

## rocprofiler-compute (Omniperf)

High-level performance analysis tool for AMD Instinct GPUs. Previously known as Omniperf, now part of rocprofiler-compute.

**Key features:**
- Web-based UI for analysis
- Detailed metrics and recommendations
- Roofline analysis
- Memory hierarchy analysis
- Workload characterization
- Built on top of rocprofiler-sdk

**Workflow:**
```bash
# Profile application
omniperf profile -n myapp -- ./myprogram

# Analyze results
omniperf analyze -p workloads/myapp

# Launch web interface
omniperf analyze -p workloads/myapp --gui
```

**Repository:** [rocprofiler-compute](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute)

**Related:** [rocprofiler-sdk](#rocprofiler-sdk), [Roofline Model](#roofline-model)

## ROCm Debugger (ROCgdb)

Source-level debugger for HIP applications based on GDB.

**Key features:**
- Set breakpoints in GPU kernels
- Inspect variables on GPU
- Step through kernel execution
- Focus on specific work-items/wavefronts

**Usage:**
```bash
rocgdb ./myprogram
(gdb) break myKernel
(gdb) run
(gdb) info rocm threads
```

**Related:** [HIP](#hip-heterogeneous-compute-interface-for-portability), [Debugging](#gpu-debugging)

## AMDGPU LLVM

AMD's LLVM backend for GPU code generation.

**Key features:**
- Compiles HIP/OpenCL to GCN/CDNA ISA
- Optimization passes for AMD GPUs
- Part of ROCm compiler stack
- Supports multiple GPU architectures

**Related:** [hipcc](#hipcc), [Code Object](#code-object)

## HIP Runtime API

Low-level API for GPU memory management, kernel launching, and device control.

**Key functions:**
```cpp
hipMalloc(&ptr, size);
hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
hipLaunchKernelGGL(kernel, grid, block, shared, stream, args...);
hipStreamCreate(&stream);
hipDeviceSynchronize();
hipFree(ptr);
```

**Related:** [HIP](#hip-heterogeneous-compute-interface-for-portability), [Memory Management](#memory-management)

## hipify

Tool to automatically convert CUDA code to HIP code.

**Usage:**
```bash
# Convert CUDA file to HIP
hipify-perl cudaCode.cu > hipCode.cpp

# Or use hipify-clang for better accuracy
hipify-clang cudaCode.cu -o hipCode.cpp
```

**Key features:**
- Converts CUDA API calls to HIP equivalents
- Handles most CUDA features
- Enables easy porting from NVIDIA to AMD

**Related:** [HIP](#hip-heterogeneous-compute-interface-for-portability), [CUDA Compatibility](#cuda-compatibility)

## Streams

Sequences of GPU operations that execute in order, enabling overlap of computation and memory transfers.

**Usage:**
```cpp
hipStream_t stream;
hipStreamCreate(&stream);

hipMemcpyAsync(d_a, h_a, size, hipMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_a);
hipMemcpyAsync(h_b, d_b, size, hipMemcpyDeviceToHost, stream);

hipStreamSynchronize(stream);
hipStreamDestroy(stream);
```

**Key benefits:**
- Overlap kernel execution with memory transfers
- Concurrent kernel execution
- Improve GPU utilization

**Related:** [Async Operations](#asynchronous-operations), [Concurrent Kernels](#concurrent-kernel-execution)

## Events

Markers in GPU streams used for timing and synchronization.

**Usage:**
```cpp
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);

hipEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
hipEventRecord(stop, stream);

hipEventSynchronize(stop);
float milliseconds;
hipEventElapsedTime(&milliseconds, start, stop);
```

**Related:** [Streams](#streams), [Performance Measurement](#performance-measurement)

## Memory Management

Host-side APIs for allocating and transferring GPU memory.

**Allocation types:**
```cpp
// Device memory
hipMalloc(&d_ptr, size);

// Pinned (page-locked) host memory
hipHostMalloc(&h_ptr, size);

// Managed/Unified memory
hipMallocManaged(&m_ptr, size);

// Free memory
hipFree(d_ptr);
hipHostFree(h_ptr);
```

**Memory transfers:**
```cpp
hipMemcpy(dst, src, size, kind);
hipMemcpyAsync(dst, src, size, kind, stream);
hipMemset(ptr, value, size);
```

**Related:** [HIP Runtime API](#hip-runtime-api), [Unified Memory](#unified-memory--managed-memory)

## GPU Monitoring

Tools and techniques for monitoring GPU health, utilization, and performance.

**Tools:**
- `rocm-smi` - Command-line monitoring
- `radeontop` - Real-time GPU stats
- Prometheus exporters for datacenter monitoring

**Key metrics:**
- GPU utilization %
- Memory utilization
- Temperature
- Power consumption
- Clock frequencies

**Related:** [rocm-smi](#rocm-smi), [Performance Analysis](#performance-analysis)
