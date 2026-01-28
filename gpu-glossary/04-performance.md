# Performance

Optimization concepts, performance metrics, and analysis techniques for AMD GPUs.

## Memory Bandwidth

The rate at which data can be transferred to/from GPU memory, measured in GB/s or TB/s.

**AMD Instinct GPUs:**
- MI300X: 5.3 TB/s (HBM3)
- MI250X: 3.2 TB/s (HBM2e)
- MI210: 1.6 TB/s (HBM2e)
- MI100: 1.2 TB/s (HBM2)

**Why it matters:**
- Often the primary bottleneck in GPU applications
- Determines achievable FLOPS for memory-bound kernels
- Critical for understanding roofline performance

**Measurement:**
```cpp
// Achieved bandwidth = (bytes_read + bytes_written) / time
```

**Related:** [HBM](#hbm-high-bandwidth-memory), [Memory Coalescing](#memory-coalescing), [Roofline Model](#roofline-model)

## Compute Throughput

The computational performance of the GPU, typically measured in FLOPS (Floating Point Operations Per Second).

**Peak theoretical performance (MI300X):**
- FP64: 163.4 TFLOPS
- FP32: 163.4 TFLOPS
- FP16: 653.7 TFLOPS
- INT8: 1307 TOPS

**Factors affecting achieved performance:**
- Memory bandwidth limitations
- Instruction mix (ALU vs memory operations)
- Occupancy and wavefront scheduling
- Divergence and control flow

**Related:** [Matrix Core Engine](#matrix-core-engine), [Roofline Model](#roofline-model)

## Roofline Model

A visual performance model that shows the achievable performance as a function of arithmetic intensity.

**Key concepts:**
- **Arithmetic Intensity**: FLOPS per byte of memory traffic
- **Compute Bound**: Limited by ALU throughput (flat part of roofline)
- **Memory Bound**: Limited by memory bandwidth (sloped part of roofline)

**Formula:**
```
Achievable FLOPS = min(Peak FLOPS, Bandwidth × Arithmetic Intensity)
```

**Usage:**
- Identify performance bottlenecks
- Guide optimization strategy
- Understand if kernel is compute or memory bound

**Related:** [Memory Bandwidth](#memory-bandwidth), [Compute Throughput](#compute-throughput), [rocprofiler-compute](#rocprofiler-compute-omniperf)

## Occupancy

The ratio of active wavefronts to the maximum supported wavefronts per Compute Unit.

**Formula:**
```
Occupancy = Active Wavefronts / Max Wavefronts per CU
```

**Limits:**
- VGPR usage (higher usage = fewer concurrent wavefronts)
- LDS usage (shared among all active workgroups)
- Workgroup size
- Maximum wavefronts per CU (typically 32-40)

**Optimal occupancy:**
- Not always 100% - balance resource usage vs. parallelism
- Higher occupancy helps hide memory latency
- Use occupancy calculator tools

**Related:** [Wavefront](#wavefront), [VGPR](#vgpr-vector-general-purpose-register), [LDS](#lds-local-data-share)

## Latency Hiding

The technique of overlapping memory access latency with computation from other wavefronts.

**Mechanisms:**
- High occupancy (more wavefronts to switch between)
- Independent instruction scheduling
- Async memory operations
- Instruction-level parallelism

**Key principle:** While one wavefront waits for memory, others compute.

**Related:** [Occupancy](#occupancy), [Memory Latency](#memory-latency)

## Memory Latency

The time delay between requesting data from memory and receiving it.

**Typical latencies (cycles):**
- Registers/VGPRs: < 1
- LDS: ~25
- L1 Cache: ~50
- L2 Cache: ~150
- HBM (global memory): ~300-400

**Mitigation strategies:**
- Increase occupancy to hide latency
- Use LDS for frequently accessed data
- Prefetch data when possible
- Optimize memory access patterns

**Related:** [Memory Hierarchy](#memory-hierarchy), [Latency Hiding](#latency-hiding)

## Wave Occupancy

The number of active wavefronts scheduled on a Compute Unit at any given time.

**Measurement:**
```bash
# Using rocprofv3 with PMC counter collection
rocprofv3 --pmc --counter GRBM_GUI_ACTIVE,SQ_WAVES ./myapp

# Or collect occupancy-related counters
rocprofv3 --pmc --counter SQ_WAVE_CYCLES,SQ_BUSY_CYCLES ./myapp
```

**Impact on performance:**
- Low occupancy → poor latency hiding → memory-bound performance
- High VGPR/LDS usage → fewer concurrent wavefronts
- Small workgroups → wasted CU capacity

**Related:** [Occupancy](#occupancy), [Compute Unit](#compute-unit-cu), [rocprofiler-sdk](#rocprofiler-sdk)

## Memory Coalescing

Combining multiple memory accesses from a wavefront into fewer transactions.

**Best practices:**
```cpp
// Good: Sequential access (coalesced)
float val = input[threadIdx.x];

// Bad: Strided access (partially coalesced)
float val = input[threadIdx.x * stride];

// Bad: Random access (uncoalesced)
float val = input[indices[threadIdx.x]];
```

**Impact:**
- Coalesced: 1-2 transactions per wavefront memory access
- Uncoalesced: Up to 64 transactions (one per work-item)
- Can reduce effective bandwidth by 10-30x

**Related:** [Memory Bandwidth](#memory-bandwidth), [Wavefront](#wavefront)

## Arithmetic Intensity

The ratio of arithmetic operations to memory operations, measured in FLOPS/byte.

**Formula:**
```
Arithmetic Intensity = FLOPS / Bytes Transferred
```

**Examples:**
- Vector addition: ~0.125 FLOPS/byte (very low)
- Matrix multiplication (large): ~100+ FLOPS/byte (high)
- Convolution: 10-50 FLOPS/byte (medium-high)

**Importance:**
- Low intensity → memory bound
- High intensity → compute bound
- Guides optimization strategy

**Related:** [Roofline Model](#roofline-model), [Memory Bandwidth](#memory-bandwidth)

## Kernel Fusion

Combining multiple kernels into a single kernel to reduce memory traffic and kernel launch overhead.

**Benefits:**
- Fewer memory roundtrips (write then read intermediate data)
- Reduced kernel launch overhead
- Better cache utilization
- Improved arithmetic intensity

**Example:**
```cpp
// Before: Two kernels
kernel1<<<grid, block>>>(input, temp);
kernel2<<<grid, block>>>(temp, output);

// After: Fused kernel
fusedKernel<<<grid, block>>>(input, output);
```

**Related:** [Memory Bandwidth](#memory-bandwidth), [Kernel](#kernel)

## Concurrent Kernel Execution

Running multiple kernels simultaneously on the same GPU using different Compute Units or Async Compute Engines.

**Requirements:**
- Sufficient free resources (CUs, memory)
- Different streams
- Hardware support (Async Compute Engines)

**Benefits:**
- Better GPU utilization
- Hide small kernel overhead
- Overlap independent work

**Usage:**
```cpp
hipStream_t stream1, stream2;
hipStreamCreate(&stream1);
hipStreamCreate(&stream2);

kernel1<<<grid1, block1, 0, stream1>>>(args1);
kernel2<<<grid2, block2, 0, stream2>>>(args2);
```

**Related:** [Streams](#streams), [Async Compute Engines](#async-compute-engines-ace)

## Bank Conflicts (LDS)

When multiple work-items in a wavefront access different words in the same LDS memory bank, causing serialization.

**LDS organization:**
- Typically 32 banks
- 4-byte words
- Banks assigned round-robin

**Avoiding conflicts:**
```cpp
// Bad: All threads access same bank
__shared__ float data[64];
float val = data[threadIdx.x * 32]; // Conflict!

// Good: Sequential access to different banks
float val = data[threadIdx.x]; // No conflict
```

**Related:** [LDS](#lds-local-data-share), [Performance Optimization](#performance-optimization-strategies)

## Instruction-Level Parallelism (ILP)

Executing multiple independent instructions concurrently within a single thread.

**Techniques:**
- Loop unrolling
- Independent variable computation
- Multiple operations per iteration

**Example:**
```cpp
// Low ILP
for (int i = 0; i < n; i++) {
    sum += data[i];
}

// Higher ILP
for (int i = 0; i < n; i += 4) {
    sum0 += data[i];
    sum1 += data[i+1];
    sum2 += data[i+2];
    sum3 += data[i+3];
}
sum = sum0 + sum1 + sum2 + sum3;
```

**Related:** [Latency Hiding](#latency-hiding), [Optimization](#performance-optimization-strategies)

## Register Pressure

The demand for VGPRs by a kernel, which can limit occupancy if too high.

**Impact:**
- High VGPR usage → fewer active wavefronts per CU
- Compiler may spill to memory (very slow)
- Reduces occupancy and performance

**Management:**
- Reduce variable lifetimes
- Use compiler flags to limit registers
- Manually optimize hot loops

**Check usage:**
```bash
# Compile and check resource usage
hipcc --offload-arch=gfx90a -c kernel.cpp
llvm-objdump --disassemble kernel.o | grep vgpr
```

**Related:** [VGPR](#vgpr-vector-general-purpose-register), [Occupancy](#occupancy)

## Divergence Overhead

Performance cost when threads in a wavefront take different execution paths.

**Causes:**
- Conditional branches (`if/else`)
- Data-dependent loops
- Early thread exit

**Impact:**
- Both paths execute serially
- Effective parallelism reduced
- Can halve performance or worse

**Mitigation:**
```cpp
// Try to structure code to minimize divergence
// Use predication instead of branching when possible
result = condition ? value1 : value2; // Better than if/else
```

**Related:** [Wave Divergence](#wave-divergence), [Wavefront](#wavefront)

## Memory Prefetching

Loading data into cache or LDS before it's needed to hide memory latency.

**Techniques:**
- Manual prefetch to LDS
- Software pipelining
- Compiler-assisted prefetching

**Example (manual LDS prefetch):**
```cpp
__shared__ float tile[TILE_SIZE];

// Prefetch next tile while processing current
tile[lid] = input[nextTileOffset + lid];
__syncthreads();

// Process current tile
// ...
```

**Related:** [LDS](#lds-local-data-share), [Latency Hiding](#latency-hiding)

## Async Copy / DMA

Hardware-accelerated asynchronous data movement between global memory and LDS.

**Benefits:**
- Offload copy operations from compute units
- Overlap data movement with computation
- Reduce VGPR pressure

**Usage (ROCm 5.0+):**
```cpp
// Async copy from global to LDS
__builtin_amdgcn_global_load_lds(...);
```

**Related:** [LDS](#lds-local-data-share), [Memory Bandwidth](#memory-bandwidth)

## Mixed Precision Training

Using lower precision (FP16, BF16, FP8) for some operations to improve performance while maintaining model accuracy.

**Precision types:**
- **FP32**: Standard precision
- **FP16**: Half precision (2x faster, less memory)
- **BF16**: Brain Float 16 (better range than FP16)
- **FP8**: 8-bit float (CDNA 3, 8x faster)

**Typical strategy:**
- Forward/backward pass: FP16/BF16
- Weight updates: FP32
- Activations: Mixed

**Hardware support:**
- MI300X: FP8, BF16, FP16
- MI250X: BF16, FP16
- MI100+: FP16

**Related:** [Matrix Core Engine](#matrix-core-engine), [Deep Learning](#deep-learning-optimization)

## FLOPS Utilization

The percentage of peak theoretical FLOPS achieved by an application.

**Formula:**
```
Utilization = (Achieved FLOPS / Peak FLOPS) × 100%
```

**Typical ranges:**
- Memory-bound kernels: 5-20%
- Compute-bound kernels: 30-60%
- Highly optimized (GEMM): 70-95%

**Improving utilization:**
- Increase arithmetic intensity
- Reduce memory bottlenecks
- Use specialized units (Matrix Cores)
- Optimize for occupancy

**Related:** [Compute Throughput](#compute-throughput), [Roofline Model](#roofline-model)

## Multi-GPU Scaling

Performance improvements from using multiple GPUs together.

**Scaling efficiency:**
```
Efficiency = (Speedup / Number of GPUs) × 100%
```

**Factors affecting scaling:**
- Inter-GPU communication overhead
- Data transfer bandwidth
- Workload parallelizability
- Synchronization frequency

**Technologies:**
- RCCL for collective operations
- Infinity Fabric for high-bandwidth links
- Peer-to-peer memory access

**Related:** [RCCL](#rccl-rocm-communication-collectives-library), [Infinity Fabric](#infinity-fabric)

## Performance Counters

Hardware metrics exposed by AMD GPUs for detailed performance analysis.

**Categories:**
- Wavefront execution metrics
- Memory subsystem (L1, L2, HBM)
- Compute unit utilization
- Instruction mix
- Stalls and bottlenecks

**Access via:**
- [rocprofiler-sdk](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk) - Command-line tool and API
- [rocprofiler-compute](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute) - High-level analysis (formerly Omniperf)
- rocprofiler-sdk API for programmatic access

**Related:** [rocprofiler-sdk](#rocprofiler-sdk), [rocprofiler-compute](#rocprofiler-compute-omniperf)

## Profiling and Analysis

The process of measuring and understanding GPU application performance.

**Tools:**
1. **[rocprofiler-sdk](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk)**: Low-level profiling, hardware counters, API tracing
2. **[rocprofiler-compute](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute)**: High-level analysis with web UI (formerly Omniperf)
3. **Manual timing**: Using HIP events

**Methodology:**
1. Identify hotspots (most time-consuming kernels)
2. Analyze bottlenecks (compute vs memory bound)
3. Check occupancy and resource usage
4. Optimize critical kernels
5. Measure improvement

**Related:** [rocprofiler-sdk](#rocprofiler-sdk), [rocprofiler-compute](#rocprofiler-compute-omniperf)

## Performance Optimization Strategies

High-level approaches to improving GPU application performance.

**Common strategies:**

1. **Memory optimization**
   - Coalesce memory accesses
   - Use LDS for frequently accessed data
   - Minimize global memory traffic
   - Prefetch data

2. **Compute optimization**
   - Maximize occupancy
   - Reduce divergence
   - Use appropriate precision (FP16/BF16)
   - Leverage Matrix Cores for GEMM

3. **Algorithmic optimization**
   - Kernel fusion
   - Tiling/blocking
   - Increase arithmetic intensity
   - Overlap computation and communication

4. **Launch configuration**
   - Tune workgroup size
   - Balance CU utilization
   - Use streams for concurrency

**Related:** [Roofline Model](#roofline-model), [Profiling](#profiling-and-analysis)
