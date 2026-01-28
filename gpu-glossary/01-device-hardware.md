# Device Hardware

Physical components and architecture of AMD GPUs.

## Compute Unit (CU)

The fundamental building block of AMD GPU architecture. A Compute Unit contains multiple Stream Processors, local data share (LDS) memory, L1 cache, and scheduling hardware. It's analogous to NVIDIA's Streaming Multiprocessor (SM).

**Key characteristics:**
- Contains 64 Stream Processors (in RDNA/CDNA architectures)
- Has dedicated LDS (Local Data Share) memory (64 KB typical)
- Includes scalar and vector ALUs
- Multiple CUs per GPU (e.g., MI300X has 304 CUs)

**Related:** [Stream Processor](#stream-processor), [Workgroup](#workgroup), [LDS](#lds-local-data-share)

## Stream Processor (SP)

The individual processing core within a Compute Unit that executes individual threads. Also called a "shader core" or "SIMD lane."

**Key characteristics:**
- Executes arithmetic and logic operations
- Works in lockstep with other SPs in a wavefront
- Each CU typically contains 64 SPs
- Supports FP32, FP64, INT operations

**Related:** [Compute Unit](#compute-unit-cu), [Wavefront](#wavefront)

## Matrix Core Engine

Specialized hardware accelerators for matrix multiplication operations, crucial for AI and deep learning workloads. Introduced in CDNA architecture.

**Key characteristics:**
- Accelerates GEMM (General Matrix Multiply) operations
- Supports multiple data types: FP64, FP32, FP16, BF16, INT8
- Dramatically faster than standard compute for matrix operations
- CDNA 3 (MI300X) supports FP8 formats

**Related:** [rocBLAS](#rocblas), [Tensor Operations](#tensor-operations)

## Graphics Compute Die (GCD)

A chiplet in multi-die GPU designs that contains the compute resources. The MI250/MI250X uses two GCDs connected via Infinity Fabric.

**Key characteristics:**
- Contains Compute Units, memory controllers, and cache
- Multiple GCDs can be connected for higher performance
- MI250X has 2 GCDs (110 CUs each)
- Enables modular GPU design

**Related:** [Infinity Fabric](#infinity-fabric), [Chiplet Architecture](#chiplet-architecture)

## Infinity Fabric

AMD's high-bandwidth, low-latency interconnect technology that connects different components within and between GPUs.

**Key characteristics:**
- Connects dies within a single package (e.g., MI250X GCDs)
- Enables GPU-to-GPU communication
- Supports coherent memory access
- Critical for multi-GPU scaling

**Bandwidth:**
- MI300X: Up to 896 GB/s per link
- MI250X: Up to 200 GB/s per link

**Related:** [GCD](#graphics-compute-die-gcd), [Multi-GPU](#multi-gpu-communication)

## HBM (High Bandwidth Memory)

Stacked memory technology providing extremely high bandwidth for GPU operations. AMD Instinct GPUs use HBM2, HBM2e, or HBM3.

**Generations:**
- **HBM2**: MI100 (1.2 TB/s), MI50/MI60 (1 TB/s)
- **HBM2e**: MI210 (1.6 TB/s), MI250/MI250X (3.2 TB/s)
- **HBM3**: MI300X (5.3 TB/s), MI300A (5.3 TB/s)

**Key characteristics:**
- Stacked directly on GPU package
- Multiple stacks per GPU
- Much higher bandwidth than GDDR memory
- Lower power consumption per GB transferred

**Related:** [Memory Bandwidth](#memory-bandwidth), [Memory Coalescing](#memory-coalescing)

## LDS (Local Data Share)

Fast, low-latency memory shared among all work-items (threads) within a workgroup. Similar to CUDA's "shared memory."

**Key characteristics:**
- 64 KB per Compute Unit (typical)
- Programmer-managed scratch pad memory
- Much faster than global memory (HBM)
- Used for inter-thread communication within a workgroup
- Critical for performance optimization

**Usage:**
```cpp
// HIP example
__shared__ float sharedData[256];
```

**Related:** [Compute Unit](#compute-unit-cu), [Workgroup](#workgroup), [Memory Hierarchy](#memory-hierarchy)

## L1 Cache

First-level cache within each Compute Unit, providing fast access to frequently used data.

**Key characteristics:**
- 16 KB per CU (typical in CDNA)
- Lowest latency memory after registers and LDS
- Automatically managed by hardware
- Separate from or combined with LDS depending on architecture

**Related:** [L2 Cache](#l2-cache), [Compute Unit](#compute-unit-cu)

## L2 Cache

Second-level cache shared across all Compute Units on the GPU.

**Key characteristics:**
- MI300X: 256 MB
- MI250X: 16 MB (8 MB per GCD)
- MI210: 8 MB
- Shared by all CUs
- Reduces global memory traffic

**Related:** [L1 Cache](#l1-cache), [Memory Hierarchy](#memory-hierarchy)

## Register File

Fast storage for thread-local variables. Each work-item (thread) has its own register allocation.

**Key characteristics:**
- Fastest memory in the hierarchy
- Limited quantity per thread
- Allocated statically per kernel
- Register pressure affects occupancy

**Related:** [Occupancy](#occupancy), [Wave Occupancy](#wave-occupancy)

## VGPR (Vector General Purpose Register)

Vector registers used for per-thread computation in AMD GPUs.

**Key characteristics:**
- Each work-item has exclusive access to its VGPRs
- CDNA 2: Up to 512 VGPRs per work-item
- Used for arithmetic operations
- High VGPR usage can limit occupancy

**Related:** [SGPR](#sgpr-scalar-general-purpose-register), [Register File](#register-file)

## SGPR (Scalar General Purpose Register)

Scalar registers shared across all lanes in a wavefront, used for uniform values.

**Key characteristics:**
- Shared by entire wavefront
- Used for addresses, loop counters, constants
- More efficient than VGPRs for uniform data
- CDNA 2: 102 SGPRs available

**Related:** [VGPR](#vgpr-vector-general-purpose-register), [Wavefront](#wavefront)

## Memory Hierarchy

The multi-level structure of memory in AMD GPUs, from fastest/smallest to slowest/largest:

1. **Registers** (VGPRs/SGPRs) - Sub-nanosecond latency
2. **LDS (Local Data Share)** - ~25 cycles latency, 64 KB per CU
3. **L1 Cache** - ~50 cycles, 16 KB per CU
4. **L2 Cache** - ~150 cycles, 8-256 MB total
5. **HBM (Global Memory)** - ~300-400 cycles, 32-192 GB total

**Related:** [LDS](#lds-local-data-share), [HBM](#hbm-high-bandwidth-memory), [Registers](#register-file)

## Chiplet Architecture

Modern AMD GPU design approach using multiple smaller dies (chiplets) connected together instead of one monolithic die.

**Benefits:**
- Better manufacturing yield
- Modularity and scalability
- Mix different process nodes (e.g., compute on 5nm, I/O on 6nm)

**Examples:**
- MI300X: 8 XCD (compute) + 4 IOD chiplets
- MI250X: 2 GCD chiplets

**Related:** [GCD](#graphics-compute-die-gcd), [Infinity Fabric](#infinity-fabric)

## Command Processor

Hardware unit responsible for reading command buffers and dispatching work to the GPU.

**Key functions:**
- Parses command buffers from CPU
- Manages kernel dispatch queues
- Coordinates work distribution to CUs
- Handles synchronization primitives

**Related:** [Kernel Dispatch](#kernel-dispatch), [HSA Queue](#hsa-queue)

## Async Compute Engines (ACE)

Independent command processors that enable concurrent execution of multiple kernels and graphics/compute overlap.

**Key characteristics:**
- Multiple ACEs per GPU (typically 8+)
- Enable kernel concurrency
- Support overlapping compute and graphics
- Each ACE has independent command queues

**Related:** [Concurrent Kernels](#concurrent-kernel-execution), [Command Processor](#command-processor)
