const express = require('express');
const path = require('path');
const os = require('os');
const glossary = require('./utils/glossary-loader');
const apiRoutes = require('./routes/api');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 8080;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Security headers
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  next();
});

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));

// Serve diagram files from gpu-glossary/diagrams
app.use('/diagrams', express.static(path.join(__dirname, '..', 'gpu-glossary', 'diagrams')));

// API routes
app.use('/api', apiRoutes);

// Serve index.html for root path
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Not found' });
});

// Error handler
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// Helper function to get network IP address
function getNetworkIP() {
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name]) {
      // Skip internal (loopback) and non-IPv4 addresses
      if (iface.family === 'IPv4' && !iface.internal) {
        return iface.address;
      }
    }
  }
  return null;
}

// Load glossary data before starting server
console.log('AMD GPU Glossary Server');
console.log('======================');

try {
  glossary.load();

  // Start server on all interfaces
  app.listen(PORT, '0.0.0.0', () => {
    const networkIP = getNetworkIP();

    console.log(`\nServer running at:`);
    console.log(`  - Local:   http://localhost:${PORT}`);
    if (networkIP) {
      console.log(`  - Network: http://${networkIP}:${PORT}`);
    }
    console.log(`\nOpen your browser to view the glossary\n`);
  });
} catch (error) {
  console.error('Failed to start server:', error);
  process.exit(1);
}
