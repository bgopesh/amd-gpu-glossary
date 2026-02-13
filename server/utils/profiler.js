const { exec } = require('child_process');
const { promisify } = require('util');
const path = require('path');
const fs = require('fs').promises;

const execAsync = promisify(exec);

// Common performance counters grouped by category
const COMMON_COUNTERS = {
  'Compute': [
    'SQ_WAVES',
    'SQ_INSTS_VALU',
    'SQ_INSTS_SALU',
    'SQ_INSTS_MFMA',
    'GRBM_GUI_ACTIVE',
    'CPC_ME1_BUSY_FOR_PACKET_DECODE'
  ],
  'Memory': [
    'TCP_TOTAL_CACHE_ACCESSES',
    'TCP_TCC_READ_REQ_sum',
    'TCP_TCC_WRITE_REQ_sum',
    'TCC_HIT_sum',
    'TCC_MISS_sum',
    'TCC_EA_RDREQ_32B_sum',
    'TCC_EA_WRREQ_sum'
  ],
  'LDS': [
    'SQ_LDS_BANK_CONFLICT',
    'SQ_LDS_ADDR_CONFLICT',
    'SQ_LDS_UNALIGNED_STALL'
  ],
  'Wavefront': [
    'SQ_WAVES',
    'SQ_WAVE_CYCLES',
    'SQ_WAIT_INST_LDS',
    'SQ_ACTIVE_INST_VALU'
  ]
};

class Profiler {
  constructor() {
    this.samplesDir = path.join(__dirname, '../../samples/hip');
  }

  /**
   * Check if rocprofv3 is available
   */
  async checkStatus() {
    try {
      const { stdout, stderr } = await execAsync('which rocprofv3');
      const rocprofPath = stdout.trim();

      if (!rocprofPath) {
        return {
          available: false,
          error: 'rocprofv3 not found in PATH'
        };
      }

      // Try to get version
      try {
        const { stdout: versionOut } = await execAsync('rocprofv3 --version');
        return {
          available: true,
          path: rocprofPath,
          version: versionOut.trim()
        };
      } catch (e) {
        return {
          available: true,
          path: rocprofPath,
          version: 'Unknown'
        };
      }
    } catch (error) {
      return {
        available: false,
        error: error.message
      };
    }
  }

  /**
   * Get available performance counters from rocprofv3
   */
  async getAvailableCounters() {
    try {
      // First check if rocprofv3 is available
      const status = await this.checkStatus();
      if (!status.available) {
        // Return common counters as fallback
        return {
          available: false,
          counters: COMMON_COUNTERS,
          message: 'Using common counter list (rocprofv3 not available)'
        };
      }

      // Try to get counters from rocprofv3
      try {
        const { stdout } = await execAsync('rocprofv3 --list-counters 2>/dev/null || rocprofv3-avail pmc 2>/dev/null', {
          timeout: 10000
        });

        // Parse counter output
        const lines = stdout.split('\n');
        const counters = {};
        let currentCategory = 'General';

        lines.forEach(line => {
          line = line.trim();
          if (!line || line.startsWith('#')) return;

          // Check if it's a category header
          if (line.match(/^[A-Z][A-Za-z\s]+:$/)) {
            currentCategory = line.replace(':', '').trim();
            counters[currentCategory] = [];
          } else if (line.match(/^[A-Z_]+/)) {
            // It's a counter name
            const counterName = line.split(/\s+/)[0];
            if (!counters[currentCategory]) {
              counters[currentCategory] = [];
            }
            counters[currentCategory].push(counterName);
          }
        });

        // If we got counters, return them; otherwise use common counters
        if (Object.keys(counters).length > 0) {
          return {
            available: true,
            counters
          };
        }
      } catch (e) {
        // Fall through to return common counters
      }

      // Return common counters as fallback
      return {
        available: true,
        counters: COMMON_COUNTERS,
        message: 'Using common counter list'
      };
    } catch (error) {
      return {
        available: false,
        counters: COMMON_COUNTERS,
        error: error.message
      };
    }
  }

  /**
   * Get list of sample applications
   */
  getSampleApplications() {
    return [
      {
        id: 'vector_add',
        name: 'Vector Addition',
        description: 'Simple vector addition kernel',
        executable: path.join(this.samplesDir, 'vector_add'),
        source: path.join(this.samplesDir, 'vector_add.cpp'),
        defaultArgs: '1000000'
      },
      {
        id: 'matrix_multiply',
        name: 'Matrix Multiplication',
        description: 'Tiled matrix multiplication with shared memory',
        executable: path.join(this.samplesDir, 'matrix_multiply'),
        source: path.join(this.samplesDir, 'matrix_multiply.cpp'),
        defaultArgs: '1024'
      }
    ];
  }

  /**
   * Build sample application if needed
   */
  async buildSampleIfNeeded(appId) {
    const apps = this.getSampleApplications();
    const app = apps.find(a => a.id === appId);

    if (!app) {
      throw new Error(`Unknown application: ${appId}`);
    }

    // Check if executable exists
    try {
      await fs.access(app.executable);
      return app.executable;
    } catch {
      // Need to build
      console.log(`Building ${appId}...`);
      const { stdout, stderr } = await execAsync(`make ${appId}`, {
        cwd: this.samplesDir,
        timeout: 60000
      });

      if (stderr && !stderr.includes('warning')) {
        console.error('Build warnings/errors:', stderr);
      }

      return app.executable;
    }
  }

  /**
   * Run profiling with rocprofv3
   */
  async runProfiling(options) {
    const { application, counters, customPath, appArgs } = options;

    // Determine executable path
    let execPath;
    if (customPath) {
      execPath = customPath;
    } else if (application) {
      execPath = await this.buildSampleIfNeeded(application);
    } else {
      throw new Error('No application specified');
    }

    // Build rocprofv3 command
    let cmd = 'rocprofv3';

    // Add counters if specified
    if (counters && counters.length > 0) {
      cmd += ` --pmc ${counters.join(',')}`;
    } else {
      // Use basic stats mode
      cmd += ' --stats';
    }

    // Add output format
    cmd += ' --output-format csv';
    cmd += ' --output-directory /tmp/rocprof-results';

    // Add application
    cmd += ` -- ${execPath}`;

    // Add application arguments
    if (appArgs) {
      cmd += ` ${appArgs}`;
    }

    console.log('Running profiler command:', cmd);

    try {
      const startTime = Date.now();
      const { stdout, stderr } = await execAsync(cmd, {
        timeout: 120000,  // 2 minute timeout
        maxBuffer: 10 * 1024 * 1024  // 10MB buffer
      });
      const duration = Date.now() - startTime;

      // Parse output
      const result = {
        success: true,
        duration,
        command: cmd,
        stdout,
        stderr,
        counters: {}
      };

      // Try to parse CSV output
      try {
        const csvPath = '/tmp/rocprof-results/pmc_1.csv';
        const csvContent = await fs.readFile(csvPath, 'utf-8');
        result.rawData = csvContent;
        result.counters = this.parseCSVCounters(csvContent);
      } catch (e) {
        // CSV might not exist, parse from stdout
        result.counters = this.parseTextOutput(stdout + stderr);
      }

      return result;
    } catch (error) {
      return {
        success: false,
        error: error.message,
        command: cmd,
        stdout: error.stdout || '',
        stderr: error.stderr || ''
      };
    }
  }

  /**
   * Parse counter values from CSV
   */
  parseCSVCounters(csvContent) {
    const lines = csvContent.split('\n');
    const counters = {};

    // Skip header, parse data
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      const parts = line.split(',');
      if (parts.length >= 2) {
        const counterName = parts[0].trim();
        const value = parseFloat(parts[parts.length - 1]);
        if (!isNaN(value)) {
          counters[counterName] = value;
        }
      }
    }

    return counters;
  }

  /**
   * Parse counter values from text output
   */
  parseTextOutput(output) {
    const counters = {};
    const lines = output.split('\n');

    lines.forEach(line => {
      // Match patterns like "CounterName: 12345" or "CounterName 12345"
      const match = line.match(/^([A-Z_][A-Z0-9_]+)\s*[:\s]+(\d+(?:\.\d+)?)/);
      if (match) {
        counters[match[1]] = parseFloat(match[2]);
      }
    });

    return counters;
  }
}

module.exports = new Profiler();
