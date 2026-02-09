# Understanding ISA (Instruction Set Architecture)

## Overview

Understanding the **Instruction Set Architecture (ISA)** is crucial for debugging and optimizing GPU applications. This guide helps you read and interpret AMDGCN (AMD Graphics Core Next) ISA, the low-level language that AMD GPUs actually execute.

---

## What is ISA?

**ISA** is the language of a processor (CPUs, GPUs, or FPGAs) that defines what operations the processor is capable of performing. It's the interface between hardware and software, specifying:
- Available instructions
- Register types and organization
- Memory addressing modes
- Data types and operations

**AMDGCN** is AMD's GPU-specific ISA, with different implementations for various architectures:
- **Vega™** (GCN 5th gen)
- **RDNA™** (Gaming GPUs)
- **CDNA™** (Compute GPUs like MI250X, MI300X)

---

## Computer Architecture Basics

### Data Sizes
- **Byte**: 8 bits
- **Word**: 16 bits (2 bytes)
- **Dword**: 32 bits (4 bytes)
- **Dwordx2**: 64 bits (8 bytes)
- **Dwordx4**: 128 bits (16 bytes)

Instructions like `s_load_dword`, `s_load_dwordx2`, and `s_load_dwordx4` load 4, 8, and 16 bytes respectively.

### Instruction Architecture

AMDGCN uses a **load/store architecture** (unlike x86's register-memory model):
- **Memory operations**: Move data between registers and memory
- **ALU/FPU operations**: Perform computations on data in registers

### Execute Mask

A 64-bit bitmask determining which threads in a wavefront execute an instruction:
- **1**: Thread executes the instruction
- **0**: Thread skips the instruction (masked out)

Used for conditional execution and divergent control flow.

---

## Processor Subunits

AMD GPUs work with **wavefronts** - groups of threads executing in lockstep:
- **CDNA2™**: 64 threads per wavefront
- **RDNA2™**: 32 or 64 threads per wavefront (configurable)

### SALU (Scalar Arithmetic Logic Unit)
- Operates on **one value per wavefront**
- Handles control flow (if/else statements, loops)
- Uses **scalar registers (SGPRs)**
- All threads in wavefront see same value

### VALU (Vector Arithmetic Logic Unit)
- Operates on **unique values per thread**
- All threads in wavefront execute together
- Uses **vector registers (VGPRs)**
- Each thread can have different values

### SMEM (Scalar Memory)
- Transfers data between **scalar registers** and **memory**
- Uniform loads (same address for all threads)

### VMEM (Vector Memory)
- Transfers data between **vector registers** and **memory**
- Each thread can use a **unique address**
- Supports gather/scatter operations

### LDS (Local Data Share)
- High-speed **shared memory** within a compute unit
- Analogous to CUDA shared memory
- Shared between all threads in a workgroup
- Much faster than global memory

---

## Registers

### Scalar General Purpose Registers (SGPRs)
- **32-bit** registers
- Store uniform data (same across all threads in wavefront)
- **MI200**: 800 SGPRs per VALU
- Used for addresses, loop counters, constants

### Vector General Purpose Registers (VGPRs)
- **32-bit** per thread
- Store per-thread data
- **MI200**: Up to 256 VGPRs + 256 AGPRs per thread
- Used for computation results, per-thread values

### Register Concatenation
For larger data types, registers are concatenated:
- **64-bit double**: Two consecutive 32-bit registers
- **64-bit pointer**: Two consecutive 32-bit registers
- Example: `s[4:5]` means registers s4 and s5 combined

### Special Registers
- **SCC (Scalar Condition Code)**: 1-bit result of scalar compare
- **VCC (Vector Condition Code)**: 64-bit result of vector compare
- **EXEC**: 64-bit execution mask for wavefront

---

## Instruction Naming Convention

### Prefixes
- **`s_`**: Scalar instructions (operate on SGPRs)
- **`v_`**: Vector instructions (operate on VGPRs)

### Common Instruction Types

#### Arithmetic
- `s_add_i32`: Scalar 32-bit integer addition
- `s_sub_i32`: Scalar 32-bit integer subtraction
- `v_add_i32`: Vector 32-bit integer addition
- `v_sub_i32`: Vector 32-bit integer subtraction

#### Move
- `v_mov_b32`: Move 32-bit value to vector register
- `s_mov_b64`: Move 64-bit value to scalar register

#### Compare (SOPC/VOPC)
- Format: `*_cmp_*`
- `v_cmp_gt_i32`: Vector compare greater than (32-bit integer)
- `s_cmp_eq_i32`: Scalar compare equal (32-bit integer)
- Sets SCC or VCC based on result

#### Conditionals
- `s_cbranch_execz`: Branch if execution mask is zero
- `s_cselect_b32`: Conditional select based on SCC

#### Loads/Stores
- `s_load_dword`: Load dword from memory into SGPR
- `global_load_dword`: Load dword from global memory into VGPR (per thread)
- `global_store_dword`: Store dword from VGPR to global memory

---

## Memory Hierarchy

From **fastest** to **slowest**:

1. **Registers (SGPRs/VGPRs)** - Immediate access, highest bandwidth
2. **LDS/L1 Cache** - Located in compute unit, shared by workgroup
3. **L2 Cache** - Shared between compute units
4. **HBM (Global Memory)** - High Bandwidth Memory, largest capacity

### Scratch Memory
- **Private per-thread memory** in global memory space
- Used when **register pressure** is high (too many variables)
- Slower than registers
- Uses `vm_cnt` counter, not `lgkmcnt`

---

## Wait Counters

AMDGCN uses **wait counters** to manage dependencies:

### `s_waitcnt`
Waits for outstanding memory operations to complete:
- **`lgkmcnt(N)`**: Wait for LDS/GDS/K-cache operations (scalar loads)
- **`vmcnt(N)`**: Wait for vector memory operations (global loads/stores)
- **`expcnt(N)`**: Wait for export operations

Example:
```asm
s_load_dwordx4 s[0:3], s[4:5], 0x8    ; Load pointers
s_waitcnt lgkmcnt(0)                   ; Wait for scalar load to complete
```

---

## Generating ISA

### Compile with ISA Output
```bash
hipcc -c --save-temps -g example.cpp
```

This generates:
- `*.s` file containing ISA
- `*.bc` LLVM bitcode
- `*.o` object file

### Get Resource Usage
```bash
hipcc -c example.cpp -Rpass-analysis=kernel-resource-usage
```

Output shows:
- VGPR usage
- SGPR usage
- Scratch memory usage
- LDS usage
- Occupancy

---

## ISA Example: Load/Store

### HIP Kernel
```cpp
__global__ void load_store(int n, float* in, float* out) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  out[tid] = in[tid];
}
```

### Key ISA Instructions

#### 1. Load Kernel Arguments
```asm
s_load_dword s7, s[4:5], 0x24          ; Load blockDim.x
s_load_dwordx4 s[0:3], s[4:5], 0x8     ; Load in[] and out[] pointers
```

#### 2. Wait for Scalar Loads
```asm
s_waitcnt lgkmcnt(0)                   ; Wait for scalar memory loads
```

#### 3. Calculate Thread ID
```asm
v_add_u32 v0, vcc, s6, v0              ; tid = threadIdx.x + blockIdx.x * blockDim.x
v_mul_lo_u32 v0, s7, v0                ; Multiply for offset
```

#### 4. Calculate Byte Offset
```asm
v_lshlrev_b64 v[0:1], 2, v[0:1]        ; tid * 4 (4 bytes per float)
```

#### 5. Load from Global Memory
```asm
global_load_dword v2, v[2:3], off      ; Load from in[tid]
s_waitcnt vmcnt(0)                     ; Wait for global load
```

#### 6. Store to Global Memory
```asm
global_store_dword v[0:1], v2, off     ; Store to out[tid]
```

---

## ISA Example: Conditionals

### HIP Kernel with Bounds Check
```cpp
__global__ void load_store(int n, float* in, float* out) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid < n) {
    out[tid] = in[tid];
  }
}
```

### ISA for Conditional

#### 1. Compare
```asm
v_cmp_gt_i32_e32 vcc, s1, v0           ; Compare n > tid, store in VCC
```

#### 2. Save and Update Execution Mask
```asm
s_and_saveexec_b64 s[0:1], vcc         ; Save EXEC, then EXEC = EXEC & VCC
```

#### 3. Conditional Branch
```asm
s_cbranch_execz .LBB0_2                ; Branch if all threads masked out
```

Threads with `tid >= n` are masked and skip the load/store operations.

---

## ISA Example: Scratch Memory

### Large Stack Array
```cpp
__global__ void use_scratch() {
  int y[17];  // 17 * 4 = 68 bytes
  // ... use array ...
}
```

### Resource Usage
```
ScratchSize [bytes/lane]: 96
```

### ISA Shows Buffer Operations
```asm
buffer_store_dword v0, off, s[0:3], s5 offset:0   ; Store y[0]
buffer_store_dword v1, off, s[0:3], s5 offset:4   ; Store y[1]
; ... 17 total stores ...
```

Arrays exceeding register capacity spill to scratch memory.

---

## ISA Example: Memory Offsets

### Shifted Array Access
```cpp
__global__ void shifted_copy(float* in, float* out) {
  int gid = threadIdx.x + blockDim.x * blockIdx.x;
  out[gid] = in[gid + 4];  // Shift by 4 elements
}
```

### ISA with Offset
```asm
global_load_dword v2, v[2:3], off offset:16   ; 16 = 4 shifts × 4 bytes/float
```

The offset is baked into the instruction, avoiding extra addition.

---

## Loop Unrolling and Register Pressure

### Pragma Unroll Impact

```cpp
#pragma unroll 8
for (int i = 0; i < 64; i++) {
  sum += in[i];
}
```

| Unroll Factor | VGPRs Used | Occupancy (waves/SIMD) |
|---------------|------------|------------------------|
| 8             | 21         | 8                      |
| 32            | 42         | 8                      |
| 64            | 74         | 6                      |

**Key Insight**: Aggressive unrolling increases register usage, which can **reduce occupancy** and hurt performance.

---

## Key Takeaways

1. **Kernel arguments** pass through SGPRs in AMDGCN
2. **`s_waitcnt`** manages dependencies between memory operations
3. **Large stack arrays** cause scratch memory usage (slower than registers)
4. **Aggressive loop unrolling** can reduce occupancy via register pressure
5. **ISA analysis** helps identify optimization opportunities:
   - Memory access patterns
   - Register usage
   - Branching behavior
   - Coalescing issues

---

## Tools for ISA Analysis

### 1. ROCm Compiler
Generate ISA code and resource information:
```bash
# Generate ISA (.s file)
hipcc -c --save-temps -g kernel.cpp

# View resource usage per kernel
hipcc -c kernel.cpp -Rpass-analysis=kernel-resource-usage
```

### 2. rocprofv3
Profile instruction-level execution and collect SQ (Sequencer) counters:

```bash
# Setup environment
source /opt/rocm/share/rocprofiler-sdk/setup-env.sh

# Collect instruction counts and execution metrics
rocprofv3 --pmc SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_INSTS_VMEM_RD,SQ_INSTS_VMEM_WR -- ./kernel.out

# Collect detailed instruction mix
rocprofv3 --pmc SQ_INSTS_VALU,SQ_INSTS_SALU,SQ_INSTS_FLAT,SQ_INSTS_LDS,SQ_INSTS_GDS -- ./kernel.out

# Monitor ALU stalls and memory wait cycles
rocprofv3 --pmc SQ_WAIT_INST_LDS,SQ_ACTIVE_INST_VALU,SQ_LDS_BANK_CONFLICT -- ./kernel.out

# Thread trace for instruction-level timing (MI200/MI300)
rocprofv3 --att --att-activity 8 -d isa_trace -- ./kernel.out
```

**Key ISA-related counters:**
- `SQ_INSTS_VALU`: Vector ALU instruction count
- `SQ_INSTS_SALU`: Scalar ALU instruction count
- `SQ_INSTS_VMEM_RD`: Vector memory read instructions
- `SQ_INSTS_VMEM_WR`: Vector memory write instructions
- `SQ_INSTS_FLAT`: FLAT instruction count
- `SQ_INSTS_LDS`: LDS instruction count
- `SQ_INSTS_SMEM`: Scalar memory instruction count
- `SQ_ACTIVE_INST_VALU`: Cycles spent executing VALU instructions
- `SQ_WAIT_INST_LDS`: Cycles waiting for LDS operations
- `SQ_LDS_BANK_CONFLICT`: LDS bank conflict cycles

**Example workflow:**
```bash
# 1. Compile kernel with ISA output
hipcc -c --save-temps -g kernel.cpp

# 2. Examine generated .s file
cat kernel-hip-amdgcn-amd-amdhsa-gfx90a.s

# 3. Profile instruction execution
rocprofv3 --pmc SQ_INSTS_VALU,SQ_INSTS_VMEM_RD -- ./kernel.out

# 4. For detailed instruction timing, use thread trace
rocprofv3 --att --att-activity 8 --kernel-include-regex "my_kernel" -d trace -- ./kernel.out
```

### 3. AMD GPU Architecture Manuals
- **CDNA2™ ISA Reference**: Detailed instruction specifications for MI200 series
- **CDNA3™ ISA Reference**: MI300 series specifications
- **RDNA2™/RDNA3™**: Gaming GPU architectures
- **RDNA4™**: Latest RDNA architecture

### 4. Additional Analysis Tools
- **llvm-objdump**: Disassemble compiled kernels
  ```bash
  llvm-objdump -d kernel.o
  ```
- **rocprofiler-compute**: High-level analysis with instruction mix visualization
  ```bash
  rocprof-compute profile -d results -- ./kernel.out
  rocprof-compute analyze -d results  # View instruction mix panel
  ```

---

## Related Terms

- **[Wavefront](#wavefront)**: Group of threads executing in lockstep
- **[Compute Unit](#compute-unit)**: Processing unit containing SIMD units
- **[LDS](#lds-local-data-share)**: Local shared memory
- **[VGPR/SGPR](#vgpr-vector-general-purpose-register)**: Register types
- **[Occupancy](#occupancy)**: Number of wavefronts per compute unit

---

## Further Reading

- [AMD ROCm Blog: Reading AMDGCN ISA](https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html)
- [CDNA2 ISA Architecture Reference](https://www.amd.com/en/support/documentation)
- [ROCm Documentation](https://rocm.docs.amd.com/)

---

*Understanding ISA enables developers to write more efficient GPU kernels by revealing the actual operations performed by the hardware.*
