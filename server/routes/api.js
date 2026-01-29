const express = require('express');
const router = express.Router();
const glossary = require('../utils/glossary-loader');

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

module.exports = router;
