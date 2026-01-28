# AMD GPU Glossary

A comprehensive reference guide for AMD GPU computing, covering hardware architecture, software stack, and performance optimization concepts.

## About

This glossary provides detailed explanations of AMD GPU terminology and concepts, organized into four main sections:

- **Device Hardware**: Physical GPU components and architecture (Compute Units, Stream Processors, Infinity Fabric, etc.)
- **Device Software**: Programming models and execution concepts (HIP, ROCm, Wavefronts, etc.)
- **Host Software**: Development tools, APIs, and libraries (rocm-smi, rocBLAS, MIOpen, etc.)
- **Performance**: Optimization concepts and metrics specific to AMD GPUs

## Focus

This glossary focuses on AMD's datacenter and compute GPUs, particularly the AMD Instinct series (MI300X, MI250X, MI210, MI100, etc.), which are optimized for AI, machine learning, and HPC workloads.

## Structure

The glossary content is organized to help developers understand AMD GPU computing at different levels of the stack, from low-level hardware details to high-level performance optimization strategies.

## Web Application

This repository includes a self-hosted web interface for browsing the glossary, inspired by Modal's GPU glossary design.

### Features

- **Beautiful, searchable interface** with real-time filtering
- **Intuitive ASCII diagrams** for visualizing GPU architecture and concepts
- **Category-based navigation** (Device Hardware, Software, Performance, etc.)
- **GPU specifications viewer** with detailed cards for AMD Instinct GPUs
- **Interactive modal views** for detailed term exploration
- **Responsive design** that works on all devices
- **No build tools required** - pure HTML/CSS/JavaScript

### Visual Diagrams

The glossary includes comprehensive ASCII diagrams for key concepts:
- Compute Unit architecture
- Memory hierarchy (Registers → LDS → L1 → L2 → L3 → HBM)
- Chiplet architecture (MI300X with 8 XCDs)
- Wavefront and workgroup organization
- Multi-GPU Infinity Fabric topology
- Roofline model and occupancy
- Memory coalescing patterns

See [FEATURES.md](FEATURES.md) for complete diagram list and examples.

### Quick Start

```bash
# Start with Python (recommended)
python -m http.server 8000

# Or with Node.js
npm run start:node

# Or on Windows, double-click:
start-server.bat
```

Then open http://localhost:8000 in your browser.

See [WEB_README.md](WEB_README.md) for detailed documentation, deployment options, and customization guide.

## Licensing

This project uses dual licensing:

- **Documentation content** (in the `gpu-glossary/` directory): [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
- **Code and utilities**: [MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to help improve and expand this glossary.

## Resources

- [AMD Instinct Documentation](https://www.amd.com/en/products/accelerators/instinct.html)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)

## Acknowledgments

Inspired by the [Modal GPU Glossary](https://github.com/modal-labs/gpu-glossary) for NVIDIA GPUs.
