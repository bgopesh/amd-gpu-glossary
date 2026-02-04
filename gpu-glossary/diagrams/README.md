# AMD GPU Glossary Diagrams

This directory contains technical SVG diagrams for the AMD GPU Glossary. All diagrams are scalable vector graphics optimized for technical documentation.

## Available Diagrams

### Hardware Architecture

1. **mi300x-architecture.svg**
   - Complete MI300X package architecture
   - 8 XCDs with 38 CUs each
   - Memory hierarchy (L1/L2/L3 caches)
   - 8 HBM3 stacks (192 GB total)
   - Infinity Fabric interconnect
   - Summary specifications

2. **compute-unit-structure.svg**
   - Internal structure of a single Compute Unit
   - 4 SIMD Units (64 lanes each)
   - Matrix Core Engine (MFMA)
   - Register files (VGPR/SGPR)
   - LDS (Local Data Share)
   - L1/L2 cache hierarchy

3. **xcd-structure.svg**
   - Single XCD (Accelerator Complex Die) chiplet structure
   - 38 active Compute Units in 8×5 grid layout
   - 4 MB L2 cache shared across all CUs
   - 4 Asynchronous Compute Engines (ACEs)
   - Infinity Fabric interconnect links
   - Shows 2 disabled CUs for yield management

4. **chiplet-architecture.svg**
   - MI300X multi-chiplet package overview
   - 8 XCD chiplets in 2×4 layout
   - Central Infinity Fabric network with 256 MB L3 cache
   - 8 HBM3 memory stacks (192 GB total)
   - Shows chiplet benefits: yield, modularity, scalability

5. **memory-hierarchy.svg**
   - Complete memory access hierarchy from fastest to slowest
   - Registers → LDS → L1 → L2 → L3 → HBM3
   - Latency, scope, and bandwidth for each level
   - Color-coded by speed (green=fast, red=slow)

### Software Execution Model

6. **wavefront-execution.svg**
   - Wavefront execution model (64 work-items in lockstep)
   - SIMT execution with instruction broadcast
   - Scalar (SGPR) vs Vector (VGPR) registers
   - Execution mask for divergence handling
   - Comparison with NVIDIA warp (32 threads)

7. **workgroup-structure.svg**
   - Workgroup structure (256 work-items = 4 wavefronts)
   - Execution on single Compute Unit
   - Shared LDS memory (64 KB)
   - Synchronization barriers

8. **kernel-grid.svg**
   - Complete kernel grid structure
   - 1024 workgroups × 256 threads = 262,144 total threads
   - Distribution across 304 CUs (MI300X)
   - Dynamic workgroup scheduling

### Performance Optimization

9. **memory-coalescing.svg**
   - Side-by-side comparison: coalesced vs uncoalesced access
   - Good: Sequential access (1-2 transactions)
   - Bad: Random/strided access (up to 64 transactions)
   - Performance impact: 10-30× bandwidth difference
   - Best practices for memory coalescing

10. **roofline-model.svg**
   - Roofline model performance analysis for MI300X
   - Memory-bound vs compute-bound regions
   - Ridge point calculation (AI ≈ 247 FLOPS/Byte)
   - Memory bandwidth limit (5.3 TB/s)
   - Compute limit (1,307 TFLOPS FP16/BF16)
   - Example workload placements

## Usage in Markdown

To reference these diagrams in markdown files:

```markdown
![MI300X Architecture](diagrams/mi300x-architecture.svg)
```

Or with HTML for more control:

```html
<img src="diagrams/mi300x-architecture.svg" alt="MI300X Architecture" width="100%">
```

## Technical Details

- **Format**: SVG (Scalable Vector Graphics)
- **Color Scheme**: Dark background optimized for technical documentation
- **AMD Branding**: Uses AMD red (#c8102e) for accents
- **Accessibility**: All text elements are selectable and searchable
- **Scalability**: Vector format ensures crisp rendering at any size

## Design Principles

1. **Color Coding**: Consistent color usage across diagrams
   - Red/Orange: AMD branding, critical paths, slow components
   - Blue: Compute resources, execution units
   - Green: Fast memory, optimized patterns
   - Purple: Caches, intermediate storage

2. **Gradients**: Used to add depth and visual hierarchy
3. **Arrows**: Show data flow and connections
4. **Labels**: Clear, concise technical labels with specifications
5. **Grouping**: Logical grouping of related components

## Converting from ASCII

These diagrams were converted from ASCII art in the original markdown files. The SVG versions provide:
- Better visual clarity
- Scalable resolution
- Professional appearance
- Color coding for better understanding
- Easier maintenance and updates

## License

These diagrams are part of the AMD GPU Glossary and follow the same licensing:
- Creative Commons Attribution 4.0 International License
- Free to use, modify, and distribute with attribution
