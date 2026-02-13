let charts = {
  timeline: null,
  duration: null,
  counters: null
};
let countersData = {};
let appsData = [];

document.addEventListener('DOMContentLoaded', async () => {
  await checkStatus();
  await loadApplications();
  await loadCounters();
});

async function checkStatus() {
  try {
    const response = await fetch('/api/profiler/status');
    const status = await response.json();

    const indicator = document.getElementById('status-indicator');
    if (status.available) {
      indicator.innerHTML = '<div class="status-indicator status-success"><span class="status-dot"></span>ROCprofv3 Available: ' + (status.version || 'Unknown') + '</div>';
    } else {
      indicator.innerHTML = '<div class="status-indicator status-error"><span class="status-dot"></span>ROCprofv3 Not Available: ' + (status.error || 'Unknown error') + '</div>';
    }
  } catch (error) {
    console.error('Error checking status:', error);
  }
}

async function loadApplications() {
  try {
    const response = await fetch('/api/profiler/applications');
    appsData = await response.json();

    const select = document.getElementById('app-select');
    select.innerHTML = '<option value="">-- Select Sample App --</option>';

    appsData.forEach(app => {
      const option = document.createElement('option');
      option.value = app.id;
      option.textContent = app.name + ' - ' + app.description;
      option.dataset.defaultArgs = app.defaultArgs;
      select.appendChild(option);
    });

    select.addEventListener('change', (e) => {
      if (e.target.value) {
        const defaultArgs = e.target.selectedOptions[0].dataset.defaultArgs;
        document.getElementById('app-args').value = defaultArgs || '';
        document.getElementById('custom-path').value = '';
      }
    });
  } catch (error) {
    console.error('Error loading applications:', error);
  }
}

async function loadCounters() {
  try {
    const response = await fetch('/api/profiler/counters');
    const data = await response.json();
    countersData = data.counters;

    const container = document.getElementById('counters-container');
    container.innerHTML = '';

    Object.entries(countersData).forEach(([category, counters]) => {
      const categoryDiv = document.createElement('div');
      categoryDiv.style.gridColumn = '1 / -1';
      categoryDiv.innerHTML = '<div class="category-header">' + category + '</div>';
      container.appendChild(categoryDiv);

      counters.forEach(counter => {
        const div = document.createElement('div');
        div.className = 'counter-checkbox';
        div.innerHTML = '<input type="checkbox" id="counter-' + counter + '" value="' + counter + '"><label for="counter-' + counter + '">' + counter + '</label>';
        container.appendChild(div);
      });
    });

    document.getElementById('select-all-counters').addEventListener('change', (e) => {
      const checkboxes = container.querySelectorAll('input[type="checkbox"]');
      checkboxes.forEach(cb => cb.checked = e.target.checked);
    });
  } catch (error) {
    console.error('Error loading counters:', error);
  }
}

async function runProfiling() {
  const runBtn = document.getElementById('run-btn');
  const runBtnText = document.getElementById('run-btn-text');
  const application = document.getElementById('app-select').value;
  const customPath = document.getElementById('custom-path').value.trim();
  const appArgs = document.getElementById('app-args').value.trim();
  const traceType = document.getElementById('trace-type').value;
  const enableSummary = document.getElementById('enable-summary').checked;
  const enableTimestamp = document.getElementById('enable-timestamp').checked;

  if (!application && !customPath) {
    alert('Please select a sample application or enter a custom path');
    return;
  }

  const selectedCounters = Array.from(
    document.querySelectorAll('#counters-container input[type="checkbox"]:checked')
  ).map(cb => cb.value);

  runBtn.disabled = true;
  runBtnText.innerHTML = '<span class="spinner"></span> Profiling...';

  try {
    const response = await fetch('/api/profiler/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        application,
        customPath: customPath || undefined,
        counters: selectedCounters,
        appArgs,
        traceType: traceType || undefined,
        enableSummary,
        enableTimestamp
      })
    });

    const result = await response.json();
    displayResults(result);
  } catch (error) {
    alert('Profiling failed: ' + error.message);
    console.error(error);
  } finally {
    runBtn.disabled = false;
    runBtnText.textContent = 'Run Profiling';
  }
}

function displayResults(result) {
  const container = document.getElementById('results-container');
  container.style.display = 'block';

  document.getElementById('command-output').textContent = result.command || 'N/A';

  const rawOutput = 'STDOUT:\n' + (result.stdout || '') + '\n\nSTDERR:\n' + (result.stderr || '');
  document.getElementById('raw-output').textContent = rawOutput;

  // Use trace data from backend if available, otherwise parse from output
  let traceData;
  if (result.traceData && result.traceData.length > 0) {
    // Combine all trace data
    traceData = {
      apiCalls: [],
      stats: {}
    };
    result.traceData.forEach(td => {
      if (td.parsed && td.parsed.length > 0) {
        traceData.apiCalls.push(...td.parsed);
      }
    });

    // Calculate stats
    if (traceData.apiCalls.length > 0) {
      const durations = traceData.apiCalls.map(c => c.duration).filter(d => d && !isNaN(d));
      if (durations.length > 0) {
        traceData.stats.totalCalls = traceData.apiCalls.length;
        traceData.stats.totalDuration = durations.reduce((a, b) => a + b, 0);
        traceData.stats.avgDuration = traceData.stats.totalDuration / durations.length;
        traceData.stats.maxDuration = Math.max(...durations);
        traceData.stats.minDuration = Math.min(...durations);
      }
    }
  } else {
    // Fallback to parsing stdout/stderr
    traceData = parseTraceData(result.stdout || '', result.stderr || '');
  }

  // Create summary statistics
  createSummary(result, traceData);

  // Create visualizations
  if (traceData.apiCalls && traceData.apiCalls.length > 0) {
    createTimelineChart(traceData);
    createDurationChart(traceData);
  } else {
    document.getElementById('tab-timeline').innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No trace data available.</p>';
    document.getElementById('tab-duration').innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No trace data available.</p>';
  }

  if (result.counters && Object.keys(result.counters).length > 0) {
    createCountersChart(result.counters);
  } else {
    document.getElementById('tab-counters').innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 2rem;">No counter data available.</p>';
  }

  container.scrollIntoView({ behavior: 'smooth' });
}

function parseTraceData(stdout, stderr) {
  const data = {
    apiCalls: [],
    stats: {}
  };

  const output = stdout + '\n' + stderr;
  const lines = output.split('\n');

  // Parse HIP/HSA API trace format
  // Example: "hipMalloc,1234567890,1234567950,60"
  lines.forEach(line => {
    // Match API call trace pattern: Name,Start,End,Duration or similar
    const traceMatch = line.match(/^(hip\w+|hsa\w+|__hip\w+)\s*[,\s]+(\d+(?:\.\d+)?)\s*[,\s]+(\d+(?:\.\d+)?)\s*[,\s]+(\d+(?:\.\d+)?)/i);
    if (traceMatch) {
      data.apiCalls.push({
        name: traceMatch[1],
        start: parseFloat(traceMatch[2]),
        end: parseFloat(traceMatch[3]),
        duration: parseFloat(traceMatch[4])
      });
    }

    // Alternative format: Name: Duration us
    const durationMatch = line.match(/^(hip\w+|hsa\w+|__hip\w+)\s*[:=]\s*(\d+(?:\.\d+)?)\s*(us|ms|ns)?/i);
    if (durationMatch) {
      let duration = parseFloat(durationMatch[2]);
      const unit = durationMatch[3];
      // Convert to microseconds
      if (unit === 'ms') duration *= 1000;
      else if (unit === 'ns') duration /= 1000;

      data.apiCalls.push({
        name: durationMatch[1],
        duration: duration
      });
    }
  });

  // Calculate statistics
  if (data.apiCalls.length > 0) {
    const durations = data.apiCalls.map(c => c.duration).filter(d => d);
    data.stats.totalCalls = data.apiCalls.length;
    data.stats.totalDuration = durations.reduce((a, b) => a + b, 0);
    data.stats.avgDuration = data.stats.totalDuration / durations.length;
    data.stats.maxDuration = Math.max(...durations);
    data.stats.minDuration = Math.min(...durations);
  }

  return data;
}

function createSummary(result, traceData) {
  const summaryDiv = document.getElementById('summary-stats');
  let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">';

  if (traceData.stats.totalCalls) {
    html += `
      <div class="stat-card">
        <h3>Total API Calls</h3>
        <p class="stat-value">${traceData.stats.totalCalls}</p>
      </div>
      <div class="stat-card">
        <h3>Total Duration</h3>
        <p class="stat-value">${(traceData.stats.totalDuration / 1000).toFixed(2)} ms</p>
      </div>
      <div class="stat-card">
        <h3>Avg Duration</h3>
        <p class="stat-value">${traceData.stats.avgDuration.toFixed(2)} µs</p>
      </div>
      <div class="stat-card">
        <h3>Max Duration</h3>
        <p class="stat-value">${traceData.stats.maxDuration.toFixed(2)} µs</p>
      </div>
    `;
  }

  if (result.counters && Object.keys(result.counters).length > 0) {
    const counterCount = Object.keys(result.counters).length;
    html += `
      <div class="stat-card">
        <h3>Counters Collected</h3>
        <p class="stat-value">${counterCount}</p>
      </div>
    `;
  }

  if (result.duration) {
    html += `
      <div class="stat-card">
        <h3>Profiling Time</h3>
        <p class="stat-value">${(result.duration / 1000).toFixed(2)} s</p>
      </div>
    `;
  }

  html += '</div>';
  summaryDiv.innerHTML = html;
}

function createTimelineChart(traceData) {
  const ctx = document.getElementById('timeline-chart');

  if (charts.timeline) {
    charts.timeline.destroy();
  }

  // Group by API name and count occurrences
  const apiFrequency = {};
  traceData.apiCalls.forEach(call => {
    apiFrequency[call.name] = (apiFrequency[call.name] || 0) + 1;
  });

  const labels = Object.keys(apiFrequency).slice(0, 20); // Top 20
  const values = labels.map(label => apiFrequency[label]);

  charts.timeline = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'API Call Frequency',
        data: values,
        backgroundColor: 'rgba(0, 150, 255, 0.6)',
        borderColor: 'rgba(0, 150, 255, 1)',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: 'API Call Frequency (Top 20)',
          color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary')
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary') },
          grid: { color: getComputedStyle(document.documentElement).getPropertyValue('--border-color') }
        },
        x: {
          ticks: {
            color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary'),
            maxRotation: 45,
            minRotation: 45
          },
          grid: { color: getComputedStyle(document.documentElement).getPropertyValue('--border-color') }
        }
      }
    }
  });
}

function createDurationChart(traceData) {
  const ctx = document.getElementById('duration-chart');

  if (charts.duration) {
    charts.duration.destroy();
  }

  // Calculate average duration per API
  const apiDurations = {};
  const apiCounts = {};

  traceData.apiCalls.forEach(call => {
    if (call.duration) {
      apiDurations[call.name] = (apiDurations[call.name] || 0) + call.duration;
      apiCounts[call.name] = (apiCounts[call.name] || 0) + 1;
    }
  });

  // Calculate averages and sort by total time
  const apiStats = Object.keys(apiDurations).map(name => ({
    name,
    avgDuration: apiDurations[name] / apiCounts[name],
    totalDuration: apiDurations[name],
    count: apiCounts[name]
  })).sort((a, b) => b.totalDuration - a.totalDuration).slice(0, 15);

  const labels = apiStats.map(s => s.name);
  const avgValues = apiStats.map(s => s.avgDuration);
  const totalValues = apiStats.map(s => s.totalDuration);

  charts.duration = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Avg Duration (µs)',
          data: avgValues,
          backgroundColor: 'rgba(237, 28, 36, 0.6)',
          borderColor: 'rgba(237, 28, 36, 1)',
          borderWidth: 2,
          yAxisID: 'y'
        },
        {
          label: 'Total Duration (µs)',
          data: totalValues,
          backgroundColor: 'rgba(255, 165, 0, 0.6)',
          borderColor: 'rgba(255, 165, 0, 1)',
          borderWidth: 2,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'API Duration Analysis (Top 15 by Total Time)',
          color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary')
        }
      },
      scales: {
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          title: { display: true, text: 'Avg Duration (µs)' },
          ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary') },
          grid: { color: getComputedStyle(document.documentElement).getPropertyValue('--border-color') }
        },
        y1: {
          type: 'linear',
          display: true,
          position: 'right',
          title: { display: true, text: 'Total Duration (µs)' },
          ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary') },
          grid: { drawOnChartArea: false }
        },
        x: {
          ticks: {
            color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary'),
            maxRotation: 45,
            minRotation: 45
          }
        }
      }
    }
  });
}

function createCountersChart(counters) {
  const ctx = document.getElementById('counters-chart');

  if (charts.counters) {
    charts.counters.destroy();
  }

  const labels = Object.keys(counters);
  const values = Object.values(counters);

  charts.counters = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Counter Values',
        data: values,
        backgroundColor: 'rgba(0, 200, 83, 0.6)',
        borderColor: 'rgba(0, 200, 83, 1)',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: {
          display: true,
          text: 'Performance Counter Results',
          color: getComputedStyle(document.documentElement).getPropertyValue('--text-primary')
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary') },
          grid: { color: getComputedStyle(document.documentElement).getPropertyValue('--border-color') }
        },
        x: {
          ticks: {
            color: getComputedStyle(document.documentElement).getPropertyValue('--text-secondary'),
            maxRotation: 45,
            minRotation: 45
          },
          grid: { color: getComputedStyle(document.documentElement).getPropertyValue('--border-color') }
        }
      }
    }
  });
}

function showTab(tabName) {
  document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  document.querySelector("[onclick=\"showTab('" + tabName + "')\"]").classList.add('active');
  document.getElementById('tab-' + tabName).classList.add('active');
}

function downloadOutput() {
  const outputText = document.getElementById('raw-output').textContent;
  const blob = new Blob([outputText], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'rocprof-output-' + Date.now() + '.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
