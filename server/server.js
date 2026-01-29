const express = require('express');
const path = require('path');
const glossary = require('./utils/glossary-loader');
const apiRoutes = require('./routes/api');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3000;

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

// Load glossary data before starting server
console.log('AMD GPU Glossary Server');
console.log('======================');

try {
  glossary.load();

  // Start server
  app.listen(PORT, () => {
    console.log(`\nServer running at http://localhost:${PORT}`);
    console.log(`Open your browser to view the glossary\n`);
  });
} catch (error) {
  console.error('Failed to start server:', error);
  process.exit(1);
}
