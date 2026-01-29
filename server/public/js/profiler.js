// AMD GPU Profiler Simulation

class ProfilerApp {
  constructor() {
    this.currentTheme = 'amd-red';
    this.initializeTheme();
    this.setupTabs();
    this.setupRooflineModel();
    this.setupOccupancyCalculator();
    this.setupMetricsDashboard();
  }

  // Theme Management
  initializeTheme() {
    const savedTheme = localStorage.getItem('glossary-theme') || 'amd-red';
    this.setTheme(savedTheme);
    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) {
      themeSelect.value = savedTheme;
      themeSelect.addEventListener('change', (e) => this.setTheme(e.target.value));
    }
  }

  setTheme(theme) {
    this.currentTheme = theme;
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('glossary-theme', theme);
  }

  // Tab Management
  setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
      button.addEventListener('click', () => {
        const tabName = button.dataset.tab;
        this.switchTab(tabName);
      });
    });
  }

  switchTab(tabName) {
    // Update buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
      btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Update panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
      pane.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
  }

  // ==================== ROOFLINE MODEL ====================

  setupRooflineModel() {
    this.rooflineCanvas = document.getElementById('roofline-chart');
    this.rooflineCtx = this.rooflineCanvas.getContext('2d');

    // MI300X specs
    this.peakFLOPS = 163.4; // TFLOPS FP64
    this.memBandwidth = 5.3; // TB/s
    this.ridgePoint = this.peakFLOPS / this.memBandwidth; // ~30.8 FLOPS/Byte

    // Example workloads
    this.rooflineExamples = {
      gemm: {
        name: 'GEMM (Matrix Multiply)',
        ai: 85, // Arithmetic Intensity (FLOPS/Byte)
        perf: 150, // Achieved TFLOPS
        description: 'Large matrix multiplication (2048x2048). Compute-bound workload with high arithmetic intensity.',
        details: {
          'Type': 'Compute Bound',
          'AI': '85 FLOPS/Byte',
          'Performance': '150 TFLOPS',
          'Efficiency': '92%'
        }
      },
      conv: {
        name: '2D Convolution',
        ai: 45,
        perf: 135,
        description: 'Deep learning convolution layer. Moderately compute-bound with good cache reuse.',
        details: {
          'Type': 'Compute Bound',
          'AI': '45 FLOPS/Byte',
          'Performance': '135 TFLOPS',
          'Efficiency': '83%'
        }
      },
      attention: {
        name: 'Attention Mechanism',
        ai: 25,
        perf: 125,
        description: 'Transformer attention computation. Balanced between compute and memory operations.',
        details: {
          'Type': 'Balanced',
          'AI': '25 FLOPS/Byte',
          'Performance': '125 TFLOPS',
          'Efficiency': '76%'
        }
      },
      elementwise: {
        name: 'Element-wise Operations',
        ai: 2,
        perf: 10.6,
        description: 'Simple addition/activation. Memory-bound with minimal computation per byte.',
        details: {
          'Type': 'Memory Bound',
          'AI': '2 FLOPS/Byte',
          'Performance': '10.6 TFLOPS',
          'Efficiency': '20%'
        }
      },
      reduction: {
        name: 'Reduction',
        ai: 0.5,
        perf: 2.65,
        description: 'Sum reduction operation. Heavily memory-bound with low arithmetic intensity.',
        details: {
          'Type': 'Memory Bound',
          'AI': '0.5 FLOPS/Byte',
          'Performance': '2.65 TFLOPS',
          'Efficiency': '5%'
        }
      }
    };

    // Setup example buttons
    document.querySelectorAll('#roofline-tab .example-button').forEach(btn => {
      btn.addEventListener('click', () => {
        const exampleName = btn.dataset.example;
        this.showRooflineExample(exampleName);

        // Update active state
        document.querySelectorAll('#roofline-tab .example-button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    // Draw initial chart
    this.drawRooflineChart();
  }

  drawRooflineChart(highlightExample = null) {
    const canvas = this.rooflineCanvas;
    const ctx = this.rooflineCtx;

    // Set canvas size
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 64;
    canvas.height = 400;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 60;

    // Clear canvas
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-primary');
    ctx.fillRect(0, 0, width, height);

    // Axes
    const maxAI = 200; // Max Arithmetic Intensity
    const maxPerf = 200; // Max Performance (TFLOPS)

    // Draw grid
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border-color');
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const y = padding + (height - 2 * padding) * (1 - i / 10);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-primary');
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Draw roofline
    const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--primary-color');
    const secondaryColor = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary');

    // Memory-bound region (slope)
    ctx.strokeStyle = secondaryColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    const ridgeX = padding + (this.ridgePoint / maxAI) * (width - 2 * padding);
    const ridgeY = height - padding - (this.peakFLOPS / maxPerf) * (height - 2 * padding);
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(ridgeX, ridgeY);
    ctx.stroke();

    // Compute-bound region (flat)
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(ridgeX, ridgeY);
    ctx.lineTo(width - padding, ridgeY);
    ctx.stroke();

    // Draw example points
    ctx.fillStyle = primaryColor;
    Object.values(this.rooflineExamples).forEach(example => {
      const x = padding + (example.ai / maxAI) * (width - 2 * padding);
      const y = height - padding - (example.perf / maxPerf) * (height - 2 * padding);

      if (highlightExample && highlightExample.ai === example.ai) {
        // Highlight selected example
        ctx.fillStyle = primaryColor;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fill();

        // Draw label
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-primary');
        ctx.font = '14px sans-serif';
        ctx.fillText(example.name, x + 12, y - 5);
      } else {
        // Regular point
        ctx.fillStyle = secondaryColor;
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.globalAlpha = 1.0;
      }
    });

    // Labels
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-primary');
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText('Performance (TFLOPS)', padding + 10, 30);
    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Arithmetic Intensity (FLOPS/Byte)', 0, 0);
    ctx.restore();

    // Axis labels
    ctx.font = '12px sans-serif';
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary');
    for (let i = 0; i <= 5; i++) {
      const perf = (maxPerf / 5) * i;
      const y = height - padding - (perf / maxPerf) * (height - 2 * padding);
      ctx.fillText(perf.toFixed(0), padding - 40, y + 5);
    }
  }

  showRooflineExample(exampleName) {
    const example = this.rooflineExamples[exampleName];
    if (!example) return;

    // Update details panel
    const detailsDiv = document.getElementById('example-details');
    let html = `
      <h4>${example.name}</h4>
      <p>${example.description}</p>
      <div style="margin-top: 1rem;">
    `;

    for (const [key, value] of Object.entries(example.details)) {
      html += `
        <div class="detail-item">
          <span class="detail-label">${key}:</span>
          <span class="detail-value">${value}</span>
        </div>
      `;
    }

    html += '</div>';
    detailsDiv.innerHTML = html;

    // Redraw chart with highlight
    this.drawRooflineChart(example);
  }

  // ==================== OCCUPANCY CALCULATOR ====================

  setupOccupancyCalculator() {
    this.occupancyCanvas = document.getElementById('occupancy-gauge');
    this.occupancyCtx = this.occupancyCanvas.getContext('2d');

    // CDNA 3 limits
    this.maxWavefronts = 40;
    this.totalVGPR = 512; // KB per CU
    this.totalLDS = 64; // KB per CU
    this.wavefrontSize = 64;

    // Scenarios
    this.occupancyScenarios = {
      optimal: {
        name: 'Optimal Configuration',
        workgroupSize: 256,
        vgprPerThread: 32,
        ldsPerWorkgroup: 0,
        description: 'Well-balanced resource usage achieving 100% occupancy',
        analysis: [
          'Workgroup uses 4 wavefronts (256/64)',
          'VGPR usage: 32 KB per workgroup',
          'Can fit 10 workgroups (40 wavefronts)',
          'LDS not limiting factor'
        ]
      },
      'high-vgpr': {
        name: 'High VGPR Usage',
        workgroupSize: 256,
        vgprPerThread: 128,
        ldsPerWorkgroup: 0,
        description: 'Heavy register usage limits occupancy',
        analysis: [
          'Workgroup uses 4 wavefronts',
          'VGPR usage: 128 KB per workgroup',
          'Can only fit 4 workgroups (16 wavefronts)',
          'Occupancy limited by VGPR (40%)'
        ]
      },
      'high-lds': {
        name: 'High LDS Usage',
        workgroupSize: 256,
        vgprPerThread: 32,
        ldsPerWorkgroup: 32,
        description: 'Shared memory usage limits occupancy',
        analysis: [
          'Workgroup uses 4 wavefronts',
          'LDS usage: 32 KB per workgroup',
          'Can only fit 2 workgroups (8 wavefronts)',
          'Occupancy limited by LDS (20%)'
        ]
      },
      'large-workgroup': {
        name: 'Large Workgroup',
        workgroupSize: 512,
        vgprPerThread: 64,
        ldsPerWorkgroup: 16,
        description: 'Large workgroup reduces concurrent workgroups',
        analysis: [
          'Workgroup uses 8 wavefronts (512/64)',
          'VGPR usage: 128 KB per workgroup',
          'LDS usage: 16 KB per workgroup',
          'Can fit 4-5 workgroups (32-40 wavefronts, 80-100%)'
        ]
      },
      'small-workgroup': {
        name: 'Small Workgroup',
        workgroupSize: 64,
        vgprPerThread: 48,
        ldsPerWorkgroup: 8,
        description: 'Small workgroup allows many concurrent groups',
        analysis: [
          'Workgroup uses 1 wavefront',
          'VGPR usage: 12 KB per workgroup',
          'Can fit 40 workgroups (40 wavefronts)',
          'Achieves 100% occupancy'
        ]
      }
    };

    // Setup scenario buttons
    document.querySelectorAll('#occupancy-tab .example-button').forEach(btn => {
      btn.addEventListener('click', () => {
        const scenarioName = btn.dataset.scenario;
        this.showOccupancyScenario(scenarioName);

        // Update active state
        document.querySelectorAll('#occupancy-tab .example-button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });
  }

  showOccupancyScenario(scenarioName) {
    const scenario = this.occupancyScenarios[scenarioName];
    if (!scenario) return;

    // Calculate occupancy
    const wavefrontsPerWorkgroup = scenario.workgroupSize / this.wavefrontSize;
    const vgprPerWorkgroup = (scenario.vgprPerThread * scenario.workgroupSize) / 1024; // KB
    const maxWorkgroupsByVGPR = Math.floor(this.totalVGPR / vgprPerWorkgroup);
    const maxWorkgroupsByLDS = scenario.ldsPerWorkgroup > 0
      ? Math.floor(this.totalLDS / scenario.ldsPerWorkgroup)
      : 999;
    const maxWorkgroupsByWavefronts = Math.floor(this.maxWavefronts / wavefrontsPerWorkgroup);

    const maxWorkgroups = Math.min(maxWorkgroupsByVGPR, maxWorkgroupsByLDS, maxWorkgroupsByWavefronts);
    const activeWavefronts = maxWorkgroups * wavefrontsPerWorkgroup;
    const occupancy = (activeWavefronts / this.maxWavefronts) * 100;

    // Update parameters
    document.getElementById('param-workgroup').textContent = scenario.workgroupSize;
    document.getElementById('param-vgpr').textContent = scenario.vgprPerThread;
    document.getElementById('param-lds').textContent = `${scenario.ldsPerWorkgroup} KB`;

    // Update gauge
    this.drawOccupancyGauge(occupancy);
    document.getElementById('occupancy-value').textContent = `${occupancy.toFixed(0)}%`;

    // Update details
    const detailsDiv = document.getElementById('occupancy-details');
    let html = `
      <h4>${scenario.name}</h4>
      <p style="color: var(--text-secondary); margin-bottom: 1rem;">${scenario.description}</p>
      <ul>
        <li><span>Active Wavefronts:</span><span>${activeWavefronts}/${this.maxWavefronts}</span></li>
        <li><span>Concurrent Workgroups:</span><span>${maxWorkgroups}</span></li>
        <li><span>Wavefronts/Workgroup:</span><span>${wavefrontsPerWorkgroup}</span></li>
        <li><span>VGPR/Workgroup:</span><span>${vgprPerWorkgroup.toFixed(1)} KB</span></li>
      </ul>
      <h5 style="color: var(--primary-color); margin-top: 1.5rem; margin-bottom: 0.5rem;">Analysis:</h5>
      <ul style="list-style: disc; padding-left: 1.5rem;">
    `;

    scenario.analysis.forEach(point => {
      html += `<li style="border: none; display: list-item;">${point}</li>`;
    });

    html += '</ul>';
    detailsDiv.innerHTML = html;
  }

  drawOccupancyGauge(occupancy) {
    const canvas = this.occupancyCanvas;
    const ctx = this.occupancyCtx;

    canvas.width = 250;
    canvas.height = 250;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 80;

    // Background arc
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border-color');
    ctx.lineWidth = 20;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, 0.25 * Math.PI);
    ctx.stroke();

    // Occupancy arc
    const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--primary-color');
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 20;
    ctx.beginPath();
    const endAngle = 0.75 * Math.PI + (1.5 * Math.PI * occupancy / 100);
    ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, endAngle);
    ctx.stroke();

    // Tick marks
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-secondary');
    ctx.lineWidth = 2;
    for (let i = 0; i <= 10; i++) {
      const angle = 0.75 * Math.PI + (1.5 * Math.PI * i / 10);
      const x1 = centerX + (radius - 15) * Math.cos(angle);
      const y1 = centerY + (radius - 15) * Math.sin(angle);
      const x2 = centerX + (radius - 5) * Math.cos(angle);
      const y2 = centerY + (radius - 5) * Math.sin(angle);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }

  // ==================== GPU METRICS DASHBOARD ====================

  setupMetricsDashboard() {
    this.metricsData = {
      utilization: [],
      bandwidth: [],
      power: [],
      temperature: []
    };

    this.currentWorkload = null;
    this.metricsInterval = null;

    // Setup workload buttons
    document.getElementById('workload-training').addEventListener('click', () => this.startWorkload('training'));
    document.getElementById('workload-inference').addEventListener('click', () => this.startWorkload('inference'));
    document.getElementById('workload-idle').addEventListener('click', () => this.startWorkload('idle'));
    document.getElementById('workload-mixed').addEventListener('click', () => this.startWorkload('mixed'));

    // Initialize charts
    this.initMetricsCharts();
  }

  initMetricsCharts() {
    ['utilization', 'bandwidth', 'power', 'temperature'].forEach(metric => {
      const canvas = document.getElementById(`chart-${metric}`);
      const ctx = canvas.getContext('2d');
      canvas.width = canvas.parentElement.clientWidth - 48;
      canvas.height = 150;
    });
  }

  startWorkload(workloadType) {
    this.currentWorkload = workloadType;

    // Update button states
    document.querySelectorAll('.workload-button').forEach(btn => {
      btn.style.backgroundColor = 'var(--bg-tertiary)';
      btn.style.borderColor = 'var(--border-color)';
    });
    document.getElementById(`workload-${workloadType}`).style.backgroundColor = 'var(--primary-color)';
    document.getElementById(`workload-${workloadType}`).style.color = 'white';

    // Clear existing interval
    if (this.metricsInterval) {
      clearInterval(this.metricsInterval);
    }

    // Reset data
    this.metricsData = {
      utilization: [],
      bandwidth: [],
      power: [],
      temperature: []
    };

    // Start updating metrics
    this.metricsInterval = setInterval(() => this.updateMetrics(), 1000);
  }

  updateMetrics() {
    if (!this.currentWorkload) return;

    // Workload profiles
    const profiles = {
      training: {
        utilization: { base: 95, variance: 5 },
        bandwidth: { base: 4800, variance: 300 },
        power: { base: 700, variance: 30 },
        temperature: { base: 75, variance: 3 }
      },
      inference: {
        utilization: { base: 70, variance: 10 },
        bandwidth: { base: 3200, variance: 400 },
        power: { base: 450, variance: 40 },
        temperature: { base: 60, variance: 4 }
      },
      idle: {
        utilization: { base: 5, variance: 3 },
        bandwidth: { base: 200, variance: 100 },
        power: { base: 80, variance: 10 },
        temperature: { base: 35, variance: 2 }
      },
      mixed: {
        utilization: { base: 60 + Math.sin(Date.now() / 5000) * 30, variance: 15 },
        bandwidth: { base: 2800 + Math.sin(Date.now() / 5000) * 1200, variance: 300 },
        power: { base: 500 + Math.sin(Date.now() / 5000) * 150, variance: 30 },
        temperature: { base: 65 + Math.sin(Date.now() / 8000) * 8, variance: 3 }
      }
    };

    const profile = profiles[this.currentWorkload];

    // Generate new values
    const newValues = {
      utilization: Math.max(0, Math.min(100, profile.utilization.base + (Math.random() - 0.5) * profile.utilization.variance * 2)),
      bandwidth: Math.max(0, profile.bandwidth.base + (Math.random() - 0.5) * profile.bandwidth.variance * 2),
      power: Math.max(0, Math.min(750, profile.power.base + (Math.random() - 0.5) * profile.power.variance * 2)),
      temperature: Math.max(20, Math.min(95, profile.temperature.base + (Math.random() - 0.5) * profile.temperature.variance * 2))
    };

    // Update data arrays
    Object.keys(newValues).forEach(metric => {
      this.metricsData[metric].push(newValues[metric]);
      if (this.metricsData[metric].length > 30) {
        this.metricsData[metric].shift();
      }
    });

    // Update displays
    document.getElementById('value-utilization').textContent = `${newValues.utilization.toFixed(1)}%`;
    document.getElementById('value-bandwidth').textContent = `${(newValues.bandwidth / 1000).toFixed(2)} TB/s`;
    document.getElementById('value-power').textContent = `${newValues.power.toFixed(0)} W`;
    document.getElementById('value-temp').textContent = `${newValues.temperature.toFixed(1)}°C`;

    // Update charts
    this.drawMetricChart('utilization', 100, '%');
    this.drawMetricChart('bandwidth', 5300, 'GB/s');
    this.drawMetricChart('power', 750, 'W');
    this.drawMetricChart('temperature', 100, '°C');
  }

  drawMetricChart(metric, maxValue, unit) {
    const canvas = document.getElementById(`chart-${metric}`);
    const ctx = canvas.getContext('2d');
    const data = this.metricsData[metric];

    if (data.length === 0) return;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 10;

    // Clear canvas
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-primary');
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border-color');
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding + (height - 2 * padding) * i / 4;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Draw line
    const primaryColor = getComputedStyle(document.documentElement).getPropertyValue('--primary-color');
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 2;
    ctx.beginPath();

    data.forEach((value, index) => {
      const x = padding + ((width - 2 * padding) * index / (data.length - 1 || 1));
      const y = height - padding - ((height - 2 * padding) * (value / maxValue));

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Fill area under curve
    ctx.lineTo(width - padding, height - padding);
    ctx.lineTo(padding, height - padding);
    ctx.closePath();
    ctx.fillStyle = primaryColor + '20';
    ctx.fill();
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new ProfilerApp();
});
