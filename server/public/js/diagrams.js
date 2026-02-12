// AMD GPU Glossary - Interactive Diagram Renderer

class DiagramRenderer {
  constructor() {
    this.diagrams = {
      'mi300x-architecture': this.createMI300XArchitecture.bind(this),
      'compute-unit': this.createComputeUnit.bind(this),
      'memory-hierarchy': this.createMemoryHierarchy.bind(this),
      'infinity-fabric': this.createInfinityFabric.bind(this)
    };
  }

  // Replace code blocks with diagrams
  renderDiagrams() {
    const preBlocks = document.querySelectorAll('.term-body pre');

    preBlocks.forEach(pre => {
      const code = pre.querySelector('code');
      if (!code) return;

      const text = code.textContent;

      // Check if this is an MI300X architecture diagram
      if (text.includes('AMD MI300X GPU Package') && text.includes('XCD')) {
        const diagram = this.createMI300XArchitecture();
        pre.replaceWith(diagram);
      }
      // Check if this is a Compute Unit diagram
      else if (text.includes('Compute Unit (CU)') && text.includes('SIMD Unit')) {
        const diagram = this.createComputeUnit();
        pre.replaceWith(diagram);
      }
      // Check if this is a Memory Hierarchy diagram
      else if (text.includes('EXECUTION LEVEL') && text.includes('REGISTERS')) {
        const diagram = this.createMemoryHierarchy();
        pre.replaceWith(diagram);
      }
      // Check if this is an Infinity Fabric diagram
      else if (text.includes('8-GPU MI300X System') && text.includes('Full Mesh Topology')) {
        const diagram = this.createInfinityFabric();
        pre.replaceWith(diagram);
      }
    });
  }

  // MI300X Architecture Diagram
  createMI300XArchitecture() {
    const container = document.createElement('div');
    container.className = 'diagram-container architecture-diagram';
    container.innerHTML = `
      <div class="diagram-title">
        <h3>AMD MI300X GPU Architecture (CDNA 3)</h3>
        <div class="diagram-subtitle">8 XCDs × 38 CUs = 304 Total Compute Units</div>
      </div>

      <div class="gpu-package">
        <div class="hbm-outer-container">
          <div class="hbm-label-outer">HBM3 Memory - 192 GB @ 5.3 TB/s</div>

          <div class="hbm-wrapper">
            <!-- Left HBM Stacks -->
            <div class="hbm-column hbm-left">
              ${Array.from({length: 4}, (_, i) => `
                <div class="hbm-stack">
                  <div class="hbm-stack-label">Stack ${i}</div>
                  <div class="hbm-capacity">24 GB</div>
                </div>
              `).join('')}
            </div>

            <!-- XCC Container -->
            <div class="xcc-container">
              <div class="xcd-grid">
                ${this.createXCD(0)}
                ${this.createXCD(1)}
                ${this.createXCD(2)}
                ${this.createXCD(3)}
              </div>

              <div class="infinity-fabric-layer">
                <div class="fabric-label">
                  <span class="fabric-icon">⚡</span>
                  Infinity Fabric Network
                  <span class="cache-info">256 MB L3 Cache</span>
                </div>
              </div>

              <div class="xcd-grid">
                ${this.createXCD(4)}
                ${this.createXCD(5)}
                ${this.createXCD(6)}
                ${this.createXCD(7)}
              </div>
            </div>

            <!-- Right HBM Stacks -->
            <div class="hbm-column hbm-right">
              ${Array.from({length: 4}, (_, i) => `
                <div class="hbm-stack">
                  <div class="hbm-stack-label">Stack ${i + 4}</div>
                  <div class="hbm-capacity">24 GB</div>
                </div>
              `).join('')}
            </div>
          </div>
        </div>
      </div>

      <div class="specs-summary">
        <div class="spec-item">
          <div class="spec-icon">🔢</div>
          <div class="spec-details">
            <div class="spec-value">19,456</div>
            <div class="spec-label">Stream Cores</div>
          </div>
        </div>
        <div class="spec-item">
          <div class="spec-icon">⚡</div>
          <div class="spec-details">
            <div class="spec-value">163 TFLOPS</div>
            <div class="spec-label">FP64 Performance</div>
          </div>
        </div>
        <div class="spec-item">
          <div class="spec-icon">🚀</div>
          <div class="spec-details">
            <div class="spec-value">1.3 PFLOPS</div>
            <div class="spec-label">FP16 Performance</div>
          </div>
        </div>
        <div class="spec-item">
          <div class="spec-icon">💾</div>
          <div class="spec-details">
            <div class="spec-value">288 MB</div>
            <div class="spec-label">Total Cache</div>
          </div>
        </div>
      </div>
    `;
    return container;
  }

  createXCD(id) {
    return `
      <div class="xcd" data-xcd="${id}">
        <div class="xcd-header">
          <div class="xcd-title">XCD ${id}</div>
          <div class="xcd-cu-count">38 CUs</div>
        </div>
        <div class="cu-grid">
          ${Array.from({length: 9}, () => '<div class="cu-mini"></div>').join('')}
        </div>
        <div class="xcd-cache">
          <div class="cache-label">L2: 4 MB</div>
        </div>
        <div class="xcd-aces">
          <div class="ace-label">4 × ACE</div>
        </div>
      </div>
    `;
  }

  // Compute Unit Diagram
  createComputeUnit() {
    const container = document.createElement('div');
    container.className = 'diagram-container compute-unit-diagram';
    container.innerHTML = `
      <div class="diagram-title">
        <h3>Compute Unit (CU) Architecture</h3>
        <div class="diagram-subtitle">Fundamental Execution Block</div>
      </div>

      <div class="cu-structure">
        <div class="simd-section">
          <div class="section-label">SIMD Units (4×)</div>
          <div class="simd-grid">
            ${Array.from({length: 4}, (_, i) => `
              <div class="simd-unit">
                <div class="simd-header">SIMD ${i}</div>
                <div class="simd-lanes">64 lanes</div>
                <div class="simd-ops">FP32/FP64/INT</div>
              </div>
            `).join('')}
          </div>
        </div>

        <div class="matrix-core-section">
          <div class="section-label">Matrix Core Engine</div>
          <div class="matrix-cores">
            <div class="matrix-ops">
              <div class="matrix-op">FP64<br>163 TF</div>
              <div class="matrix-op">FP32<br>163 TF</div>
              <div class="matrix-op">FP16<br>1307 TF</div>
              <div class="matrix-op">FP8<br>2615 TF</div>
            </div>
          </div>
        </div>

        <div class="memory-section">
          <div class="registers">
            <div class="register-block vgpr">
              <div class="reg-icon">📝</div>
              <div class="reg-label">VGPRs</div>
              <div class="reg-size">512 KB</div>
            </div>
            <div class="register-block sgpr">
              <div class="reg-icon">📋</div>
              <div class="reg-label">SGPRs</div>
              <div class="reg-size">12.5 KB</div>
            </div>
          </div>

          <div class="lds-block">
            <div class="lds-icon">🔄</div>
            <div class="lds-label">LDS (Local Data Share)</div>
            <div class="lds-size">64 KB</div>
            <div class="lds-latency">~25 cycles</div>
          </div>

          <div class="cache-block">
            <div class="cache-icon">⚡</div>
            <div class="cache-label">L1 Vector Cache</div>
            <div class="cache-size">32 KB</div>
            <div class="cache-latency">~50 cycles</div>
          </div>
        </div>

        <div class="scheduler-section">
          <div class="scheduler-label">Scheduler & Dispatch</div>
          <div class="scheduler-features">
            <span class="feature">Wavefront Scheduling</span>
            <span class="feature">Instruction Fetch/Decode</span>
            <span class="feature">Dependency Tracking</span>
          </div>
        </div>
      </div>
    `;
    return container;
  }

  // Memory Hierarchy Diagram
  createMemoryHierarchy() {
    const container = document.createElement('div');
    container.className = 'diagram-container memory-hierarchy-diagram';
    container.innerHTML = `
      <div class="diagram-title">
        <h3>Memory Hierarchy</h3>
        <div class="diagram-subtitle">From Fastest to Slowest</div>
      </div>

      <div class="memory-pyramid">
        <div class="memory-level level-1" data-latency="< 1 cycle">
          <div class="level-content">
            <div class="level-number">1</div>
            <div class="level-info">
              <div class="level-name">Registers</div>
              <div class="level-size">VGPRs: 512 KB | SGPRs: 12.5 KB per CU</div>
            </div>
            <div class="level-latency">< 1 cycle</div>
          </div>
        </div>

        <div class="memory-level level-2" data-latency="~25 cycles">
          <div class="level-content">
            <div class="level-number">2</div>
            <div class="level-info">
              <div class="level-name">LDS (Local Data Share)</div>
              <div class="level-size">64 KB per CU</div>
            </div>
            <div class="level-latency">~25 cycles</div>
          </div>
        </div>

        <div class="memory-level level-3" data-latency="~50 cycles">
          <div class="level-content">
            <div class="level-number">3</div>
            <div class="level-info">
              <div class="level-name">L1 Cache</div>
              <div class="level-size">32 KB per CU (9.7 MB total)</div>
            </div>
            <div class="level-latency">~50 cycles</div>
          </div>
        </div>

        <div class="memory-level level-4" data-latency="~150 cycles">
          <div class="level-content">
            <div class="level-number">4</div>
            <div class="level-info">
              <div class="level-name">L2 Cache</div>
              <div class="level-size">4 MB per XCD (32 MB total)</div>
            </div>
            <div class="level-latency">~150 cycles</div>
          </div>
        </div>

        <div class="memory-level level-5" data-latency="~200 cycles">
          <div class="level-content">
            <div class="level-number">5</div>
            <div class="level-info">
              <div class="level-name">L3 Cache (Infinity Cache)</div>
              <div class="level-size">256 MB</div>
            </div>
            <div class="level-latency">~200 cycles</div>
          </div>
        </div>

        <div class="memory-level level-6" data-latency="~300-400 cycles">
          <div class="level-content">
            <div class="level-number">6</div>
            <div class="level-info">
              <div class="level-name">HBM3 (Global Memory)</div>
              <div class="level-size">192 GB @ 5.3 TB/s</div>
            </div>
            <div class="level-latency">~300-400 cycles</div>
          </div>
        </div>

        <div class="memory-level level-7" data-latency="> 500 cycles">
          <div class="level-content">
            <div class="level-number">7</div>
            <div class="level-info">
              <div class="level-name">Remote GPU Memory</div>
              <div class="level-size">Via Infinity Fabric</div>
            </div>
            <div class="level-latency">> 500 cycles</div>
          </div>
        </div>
      </div>
    `;
    return container;
  }

  // Infinity Fabric Topology Diagram
  createInfinityFabric() {
    const container = document.createElement('div');
    container.className = 'diagram-container infinity-fabric-diagram';
    container.innerHTML = `
      <div class="diagram-title">
        <h3>8-GPU Infinity Fabric Topology</h3>
        <div class="diagram-subtitle">Full-Mesh Interconnect</div>
      </div>

      <svg class="fabric-topology" viewBox="0 0 600 600" xmlns="http://www.w3.org/2000/svg">
        <!-- Connection lines -->
        <g class="connections" stroke="#ED1C24" stroke-width="2" opacity="0.3">
          ${this.createMeshConnections()}
        </g>

        <!-- GPU nodes -->
        <g class="gpu-nodes">
          ${this.createGPUNode(300, 50, 0)}
          ${this.createGPUNode(475, 150, 1)}
          ${this.createGPUNode(525, 325, 2)}
          ${this.createGPUNode(450, 500, 3)}
          ${this.createGPUNode(300, 550, 4)}
          ${this.createGPUNode(150, 500, 5)}
          ${this.createGPUNode(75, 325, 6)}
          ${this.createGPUNode(125, 150, 7)}
        </g>
      </svg>

      <div class="fabric-features">
        <div class="feature-item">
          <div class="feature-icon">🔗</div>
          <div class="feature-text">All-to-all connectivity</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">⚡</div>
          <div class="feature-text">7 links per GPU</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">🚀</div>
          <div class="feature-text">No intermediate hops</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">📊</div>
          <div class="feature-text">Optimal for collectives</div>
        </div>
      </div>
    `;
    return container;
  }

  createMeshConnections() {
    const positions = [
      [300, 50], [475, 150], [525, 325], [450, 500],
      [300, 550], [150, 500], [75, 325], [125, 150]
    ];

    let lines = '';
    for (let i = 0; i < 8; i++) {
      for (let j = i + 1; j < 8; j++) {
        lines += `<line x1="${positions[i][0]}" y1="${positions[i][1]}"
                       x2="${positions[j][0]}" y2="${positions[j][1]}"
                       class="fabric-link" />`;
      }
    }
    return lines;
  }

  createGPUNode(x, y, id) {
    return `
      <g class="gpu-node" data-gpu="${id}">
        <circle cx="${x}" cy="${y}" r="40" fill="var(--bg-secondary)"
                stroke="var(--primary-color)" stroke-width="3" class="gpu-circle"/>
        <text x="${x}" y="${y - 5}" text-anchor="middle" fill="var(--primary-color)"
              font-weight="bold" font-size="16">GPU ${id}</text>
        <text x="${x}" y="${y + 12}" text-anchor="middle" fill="var(--text-secondary)"
              font-size="12">MI300X</text>
      </g>
    `;
  }
}

// Initialize diagrams when content is loaded
document.addEventListener('DOMContentLoaded', () => {
  // Wait a bit for content to load
  setTimeout(() => {
    const renderer = new DiagramRenderer();
    renderer.renderDiagrams();
  }, 500);
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DiagramRenderer;
}
