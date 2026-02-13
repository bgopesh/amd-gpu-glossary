const express = require('express');
const router = express.Router();
const glossary = require('../utils/glossary-loader');
const profiler = require('../utils/profiler');

// GET /api/sections - Get all sections with term listings
router.get('/sections', (req, res) => {
  try {
    const sections = glossary.getSections();
    res.json(sections);
  } catch (error) {
    console.error('Error fetching sections:', error);
    res.status(500).json({ error: 'Failed to fetch sections' });
  }
});

// GET /api/terms/:slug - Get a term by slug only (searches all sections)
router.get('/terms/:slug', (req, res) => {
  try {
    // Check if slug contains a "/" - if so, treat it as sectionId/slug
    if (req.params.slug.includes('/')) {
      const term = glossary.getTerm(req.params.slug);
      if (!term) {
        return res.status(404).json({ error: 'Term not found' });
      }
      return res.json(term);
    }

    // Otherwise, search by slug across all sections
    const term = glossary.findTermBySlug(req.params.slug);

    if (!term) {
      return res.status(404).json({ error: 'Term not found' });
    }

    res.json(term);
  } catch (error) {
    console.error('Error fetching term:', error);
    res.status(500).json({ error: 'Failed to fetch term' });
  }
});

// GET /api/terms/:sectionId/:slug - Get a specific term by ID
// ID format: section-id/term-slug (e.g., "device-hardware/compute-unit-cu")
router.get('/terms/:sectionId/:slug', (req, res) => {
  try {
    const termId = `${req.params.sectionId}/${req.params.slug}`;
    const term = glossary.getTerm(termId);

    if (!term) {
      return res.status(404).json({ error: 'Term not found' });
    }

    res.json(term);
  } catch (error) {
    console.error('Error fetching term:', error);
    res.status(500).json({ error: 'Failed to fetch term' });
  }
});

// GET /api/search?q=query - Search for terms
router.get('/search', (req, res) => {
  try {
    const query = req.query.q;

    if (!query) {
      return res.json([]);
    }

    const results = glossary.search(query);
    res.json(results);
  } catch (error) {
    console.error('Error searching:', error);
    res.status(500).json({ error: 'Search failed' });
  }
});

// GET /api/specs - Get GPU specifications
router.get('/specs', (req, res) => {
  try {
    const specs = glossary.getSpecs();
    res.json(specs);
  } catch (error) {
    console.error('Error fetching specs:', error);
    res.status(500).json({ error: 'Failed to fetch GPU specifications' });
  }
});

// GET /api/overview/:sectionId - Get overview for a section
router.get('/overview/:sectionId', (req, res) => {
  try {
    const overview = glossary.getOverview(req.params.sectionId);

    if (!overview) {
      return res.status(404).json({ error: 'Overview not found' });
    }

    res.json(overview);
  } catch (error) {
    console.error('Error fetching overview:', error);
    res.status(500).json({ error: 'Failed to fetch overview' });
  }
});

// Profiler endpoints

// GET /api/profiler/counters - Get available performance counters
router.get('/profiler/counters', async (req, res) => {
  try {
    const counters = await profiler.getAvailableCounters();
    res.json(counters);
  } catch (error) {
    console.error('Error fetching counters:', error);
    res.status(500).json({ error: 'Failed to fetch available counters', details: error.message });
  }
});

// GET /api/profiler/applications - Get available sample applications
router.get('/profiler/applications', (req, res) => {
  try {
    const apps = profiler.getSampleApplications();
    res.json(apps);
  } catch (error) {
    console.error('Error fetching applications:', error);
    res.status(500).json({ error: 'Failed to fetch sample applications', details: error.message });
  }
});

// POST /api/profiler/run - Run profiling
router.post('/profiler/run', async (req, res) => {
  try {
    const { application, counters, customPath, appArgs } = req.body;

    if (!application && !customPath) {
      return res.status(400).json({ error: 'Application or custom path required' });
    }

    const result = await profiler.runProfiling({
      application,
      counters: counters || [],
      customPath,
      appArgs: appArgs || ''
    });

    res.json(result);
  } catch (error) {
    console.error('Error running profiler:', error);
    res.status(500).json({ error: 'Profiling failed', details: error.message });
  }
});

// GET /api/profiler/status - Check rocprofv3 availability
router.get('/profiler/status', async (req, res) => {
  try {
    const status = await profiler.checkStatus();
    res.json(status);
  } catch (error) {
    console.error('Error checking profiler status:', error);
    res.status(500).json({ error: 'Failed to check profiler status', details: error.message });
  }
});

module.exports = router;
