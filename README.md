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

## Web Interface

The glossary includes a self-hosted web interface for easy browsing and searching.

### Prerequisites

- Node.js (version 14 or higher)
- npm (comes with Node.js)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/amd-gpu-glossary.git
cd amd-gpu-glossary
```

2. Install dependencies:
```bash
npm install
```

### Running the Server

Start the web server:
```bash
npm start
```

The server will start on port 8080 by default. You can access it at:
- **Local:** http://localhost:8080
- **Network:** The server will display your network IP address when it starts

To use a different port, set the PORT environment variable:
```bash
PORT=3000 npm start
```

### Features

- **Split-pane layout**: Navigate sections on the left, view content on the right
- **Full-text search**: Quickly find terms across all sections
- **Multiple themes**: Choose from 5 color schemes for optimal readability
  - AMD Red (Dark) - Default AMD branding
  - Light Mode - Bright background for daylight use
  - High Contrast - Maximum accessibility
  - Blue (Dark) - Blue accent theme
  - Green (Dark) - Green accent theme
- **Cross-references**: Click related terms to navigate between concepts
- **GPU specifications**: View detailed specs for all AMD Instinct GPUs
- **ASCII diagrams**: Visual representations of architecture and concepts
- **Profiler Simulation**: Interactive tools for understanding GPU performance
  - **Roofline Model**: Visualize performance bounds and analyze kernel efficiency with realistic workload examples (GEMM, convolution, attention, etc.)
  - **Occupancy Calculator**: Calculate Compute Unit occupancy based on resource usage with pre-loaded scenarios
  - **GPU Metrics Dashboard**: Real-time simulation of GPU metrics (utilization, bandwidth, power, temperature) under different workloads

### API Endpoints

The server exposes a RESTful API:

- `GET /api/sections` - List all sections with terms
- `GET /api/terms/:slug` - Get a specific term by slug
- `GET /api/terms/:sectionId/:slug` - Get a term by section and slug
- `GET /api/search?q=query` - Search for terms
- `GET /api/specs` - Get GPU specifications

## Structure

The glossary content is organized to help developers understand AMD GPU computing at different levels of the stack, from low-level hardware details to high-level performance optimization strategies.

### Project Structure

```
amd-gpu-glossary/
├── gpu-glossary/              # Markdown content files
│   ├── 01-device-hardware.md
│   ├── 02-device-software.md
│   ├── 03-host-software.md
│   ├── 04-performance.md
│   └── amd-gpu-specs.json
├── server/                    # Web server
│   ├── server.js             # Express app
│   ├── routes/
│   │   └── api.js           # API endpoints
│   ├── utils/
│   │   └── glossary-loader.js
│   └── public/              # Frontend assets
│       ├── index.html
│       ├── css/style.css
│       └── js/app.js
└── package.json
```

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
