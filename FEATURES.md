# AMD GPU Glossary - Features & Visual Diagrams

## ASCII Diagram Examples

The glossary now includes comprehensive ASCII diagrams that make complex AMD GPU concepts intuitive and visual. Here are the diagrams included:

### Hardware Architecture Diagrams

1. **Compute Unit (CU) Architecture**
   - Shows SIMD units, Matrix Core Engine, register files, LDS, and L1 cache
   - Visualizes the internal structure of AMD's fundamental building block

2. **Memory Hierarchy**
   - Complete visual representation from registers to HBM
   - Shows latency, size, and scope at each level
   - Includes cycle counts and bandwidth information

3. **XCD (Accelerator Complex Die)**
   - Displays 38 compute units layout
   - Shows L2 cache and Asynchronous Compute Engines (ACEs)
   - Illustrates Infinity Fabric connections

4. **Chiplet Architecture (MI300X)**
   - Full GPU package view with 8 XCDs
   - Shows HBM3 stacks placement
   - Illustrates Infinity Fabric network and L3 cache

5. **Multi-GPU Topology (8-GPU Full Mesh)**
   - Infinity Fabric interconnect between 8 MI300X GPUs
   - All-to-all connectivity visualization
   - Critical for understanding multi-GPU training setups

### Software Execution Diagrams

6. **Wavefront Execution**
   - 64 work-items executing in lockstep (SIMT)
   - Shows lane structure and shared resources
   - Compares to NVIDIA's 32-thread warp

7. **Workgroup Organization**
   - Multiple wavefronts in a workgroup
   - Shared LDS memory visualization
   - Synchronization points

8. **Grid Layout**
   - Complete kernel launch structure
   - Distribution across compute units
   - Shows workgroup independence

### Performance & Optimization Diagrams

9. **Roofline Model**
   - Visual performance model
   - Shows compute-bound vs memory-bound regions
   - Ridge point calculation

10. **Occupancy Visualization**
    - Low vs high occupancy comparison
    - Wavefront utilization in a CU
    - Shows impact on latency hiding

11. **Memory Coalescing**
    - Coalesced vs uncoalesced access patterns
    - Visual bandwidth impact
    - Best practice comparison

## Web Interface Features

### Search & Navigation
- **Real-time search** across all terms, definitions, and content
- **Category tabs** for filtering (Device Hardware, Device Software, Host Software, Performance, GPU Specs)
- **Filter indicators** showing search results count
- **Keyboard shortcuts** (ESC to close modal)

### Visual Design
- **AMD-branded color scheme** (red/white)
- **Responsive grid layout** adapts to screen size
- **Interactive term cards** with hover effects
- **Smooth animations** for modal transitions
- **Professional typography** with optimized readability

### Content Display
- **ASCII diagrams** rendered with monospace font and preserved spacing
- **Code syntax highlighting** for HIP/C++ examples
- **Cross-references** between related terms
- **GPU specification cards** with detailed metrics

### GPU Specifications Viewer
Interactive cards for AMD Instinct GPUs:
- MI300X (CDNA 3)
- MI300A (CDNA 3 + Zen 4)
- MI250X/MI250/MI210 (CDNA 2)
- MI100 (CDNA 1)
- MI60/MI50 (Vega 20)

Each card displays:
- Compute units and architecture
- Memory size, type, and bandwidth
- Cache hierarchy (L1/L2/L3)
- Performance (FP64/FP32/FP16/FP8)
- TDP and process node

### Technical Features
- **Pure HTML/CSS/JavaScript** - no build tools required
- **Client-side rendering** - fast and responsive
- **Markdown parsing** - easy content updates
- **Mobile-friendly** - works on all devices
- **Printer-friendly** - clean printing support

## Diagram Benefits

The ASCII diagrams provide:

1. **Immediate Understanding**: Complex architectures become clear at a glance
2. **Memory Aid**: Visual structure helps retention
3. **Reference Tool**: Quick lookup for architecture details
4. **Teaching Resource**: Excellent for explaining concepts to others
5. **No Dependencies**: Pure text diagrams work everywhere
6. **Version Control Friendly**: Text-based, easy to diff and track changes

## Usage Examples

### For Developers
- Understand AMD GPU architecture before optimizing code
- Quick reference for memory hierarchy and bandwidth
- Visual guide to wavefront and workgroup organization

### For Students
- Learn GPU computing concepts visually
- Compare AMD and NVIDIA architectures
- Study hardware-software interaction

### For System Architects
- Understand chiplet design and scaling
- Plan multi-GPU configurations
- Reference specifications for capacity planning

### For Performance Engineers
- Visualize bottlenecks with roofline model
- Understand occupancy requirements
- Optimize memory access patterns with coalescing diagrams

## Customization

All diagrams are in plain text and can be easily modified:
- Edit markdown files to update diagrams
- Add new diagrams following the same format
- CSS controls diagram styling (font, spacing, colors)

## Accessibility

- Diagrams use Unicode box-drawing characters for compatibility
- Monospace font ensures alignment across platforms
- High contrast for readability
- Screen reader friendly with descriptive text

## Future Enhancements

Potential additions:
- More diagrams for advanced topics (cache coherency, atomic operations)
- Interactive diagrams with hover tooltips
- Animated transitions showing execution flow
- Comparison diagrams (AMD vs NVIDIA)
- Performance profiling diagrams
