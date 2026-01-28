# AMD GPU Glossary Content

This directory contains the comprehensive documentation for AMD GPU computing terminology and concepts.

## Structure

The glossary is organized into four main sections:

### 1. Device Hardware (`01-device-hardware.md`)
Physical components and architecture of AMD GPUs including:
- Compute Units and Stream Processors
- Memory hierarchy (HBM, LDS, Cache)
- Matrix Core Engines
- Infinity Fabric interconnect
- Chiplet architecture

### 2. Device Software (`02-device-software.md`)
Programming models and execution concepts:
- HIP programming language
- Kernels, Wavefronts, and Work-items
- Workgroups and synchronization
- Memory management and coalescing
- Execution model (GCN/CDNA ISA)

### 3. Host Software (`03-host-software.md`)
Development tools, APIs, and libraries:
- ROCm platform and runtime
- Math libraries (rocBLAS, rocFFT, MIOpen)
- Profiling tools (rocprofiler-sdk, rocprofiler-compute)
- Deep learning frameworks (PyTorch, TensorFlow)
- Development tools (hipcc, rocm-smi, ROCgdb)

### 4. Performance (`04-performance.md`)
Optimization concepts and performance metrics:
- Memory bandwidth and compute throughput
- Roofline model
- Occupancy and latency hiding
- Performance optimization strategies
- Profiling and analysis techniques

## GPU Specifications

The `amd-gpu-specs.json` file contains detailed specifications for AMD Instinct datacenter GPUs including:
- MI300X/MI300A (CDNA 3)
- MI250X/MI250/MI210 (CDNA 2)
- MI100 (CDNA 1)
- MI60/MI50 (Vega 20)

## Navigation

Each section contains hyperlinked terms that reference related concepts across different sections. Terms are organized alphabetically within each section for easy reference.

## Contributing

This content is designed to be comprehensive yet accessible. Contributions that improve clarity, add missing concepts, or correct errors are welcome.

## License

This documentation content is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
