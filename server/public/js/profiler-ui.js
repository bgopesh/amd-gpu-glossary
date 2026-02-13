let chart = null;
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

  const rawOutput = 'STDOUT:\\n' + (result.stdout || '') + '\\n\\nSTDERR:\\n' + (result.stderr || '');
  document.getElementById('raw-output').textContent = rawOutput;

  if (result.counters && Object.keys(result.counters).length > 0) {
    createChart(result.counters);
  } else {
    document.getElementById('tab-chart').innerHTML = '<p style="color: var(--text-secondary); text-align: center;">No counter data available. Check raw output for details.</p>';
  }

  container.scrollIntoView({ behavior: 'smooth' });
}

function createChart(counters) {
  const ctx = document.getElementById('results-chart');

  if (chart) {
    chart.destroy();
  }

  const labels = Object.keys(counters);
  const values = Object.values(counters);

  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Counter Values',
        data: values,
        backgroundColor: 'rgba(237, 28, 36, 0.6)',
        borderColor: 'rgba(237, 28, 36, 1)',
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
