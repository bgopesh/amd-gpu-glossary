const fs = require('fs');
const path = require('path');
const MarkdownIt = require('markdown-it');
const markdownItAnchor = require('markdown-it-anchor');

// Initialize markdown parser
const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true
}).use(markdownItAnchor, {
  permalink: false,
  slugify: (s) => s.toLowerCase().replace(/[^\w\s-]/g, '').replace(/\s+/g, '-')
});

// Section definitions
const SECTIONS = [
  {
    id: 'device-hardware',
    title: 'Device Hardware',
    file: '01-device-hardware.md',
    description: 'Physical components and architecture of AMD GPUs'
  },
  {
    id: 'device-software',
    title: 'Device Software',
    file: '02-device-software.md',
    description: 'Software running on GPU (kernels, ISA)'
  },
  {
    id: 'host-software',
    title: 'Host Software',
    file: '03-host-software.md',
    description: 'CPU-side software and APIs (HIP, ROCm)'
  },
  {
    id: 'performance',
    title: 'Performance',
    file: '04-performance.md',
    description: 'Optimization and profiling tools'
  }
];

class GlossaryLoader {
  constructor() {
    this.glossaryPath = path.join(__dirname, '../../gpu-glossary');
    this.sections = [];
    this.termsById = new Map();
    this.specs = null;
  }

  // Load all glossary data
  load() {
    console.log('Loading glossary data...');

    // Load each section
    for (const sectionDef of SECTIONS) {
      const section = this.loadSection(sectionDef);
      this.sections.push(section);
    }

    // Load GPU specs
    this.loadSpecs();

    console.log(`Loaded ${this.termsById.size} terms from ${this.sections.length} sections`);
  }

  // Load a single section from markdown file
  loadSection(sectionDef) {
    const filePath = path.join(this.glossaryPath, sectionDef.file);
    const content = fs.readFileSync(filePath, 'utf-8');

    // Parse terms from markdown
    const terms = this.parseTerms(content, sectionDef.id);

    return {
      id: sectionDef.id,
      title: sectionDef.title,
      description: sectionDef.description,
      terms: terms
    };
  }

  // Parse terms from markdown content
  parseTerms(content, sectionId) {
    const lines = content.split('\n');
    const terms = [];
    let currentTerm = null;
    let currentContent = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Check for ## heading (term definition)
      if (line.startsWith('## ')) {
        // Save previous term if exists
        if (currentTerm) {
          currentTerm.content = currentContent.join('\n').trim();
          currentTerm.html = md.render(currentTerm.content);
          terms.push(currentTerm);
          this.termsById.set(currentTerm.id, currentTerm);
        }

        // Start new term
        const title = line.substring(3).trim();
        const slug = this.slugify(title);
        currentTerm = {
          id: `${sectionId}/${slug}`,
          slug: slug,
          title: title,
          sectionId: sectionId
        };
        currentContent = [];
      } else if (line.startsWith('# ')) {
        // Skip # heading (section title)
        continue;
      } else if (currentTerm) {
        // Add to current term content
        currentContent.push(line);
      }
    }

    // Save last term
    if (currentTerm) {
      currentTerm.content = currentContent.join('\n').trim();
      currentTerm.html = md.render(currentTerm.content);
      terms.push(currentTerm);
      this.termsById.set(currentTerm.id, currentTerm);
    }

    return terms;
  }

  // Load GPU specifications
  loadSpecs() {
    const specsPath = path.join(this.glossaryPath, 'amd-gpu-specs.json');
    try {
      const specsData = fs.readFileSync(specsPath, 'utf-8');
      this.specs = JSON.parse(specsData);
      console.log(`Loaded specs for ${this.specs.amd_instinct_gpus.length} GPU models`);
    } catch (error) {
      console.error('Error loading GPU specs:', error.message);
      this.specs = { amd_instinct_gpus: [] };
    }
  }

  // Get all sections with term summaries
  getSections() {
    return this.sections.map(section => ({
      id: section.id,
      title: section.title,
      description: section.description,
      terms: section.terms.map(term => ({
        id: term.id,
        slug: term.slug,
        title: term.title
      }))
    }));
  }

  // Get a specific term by ID
  getTerm(termId) {
    return this.termsById.get(termId);
  }

  // Find a term by slug (searches across all sections)
  findTermBySlug(slug) {
    for (const [id, term] of this.termsById) {
      if (term.slug === slug) {
        return term;
      }
    }
    return null;
  }

  // Search terms
  search(query) {
    if (!query || query.trim().length === 0) {
      return [];
    }

    const searchLower = query.toLowerCase();
    const results = [];

    for (const [id, term] of this.termsById) {
      let score = 0;

      // Check title match
      const titleLower = term.title.toLowerCase();
      if (titleLower === searchLower) {
        score += 100; // Exact match
      } else if (titleLower.startsWith(searchLower)) {
        score += 50; // Starts with query
      } else if (titleLower.includes(searchLower)) {
        score += 25; // Contains query
      }

      // Check content match
      const contentLower = term.content.toLowerCase();
      if (contentLower.includes(searchLower)) {
        score += 10;
      }

      if (score > 0) {
        results.push({
          id: term.id,
          slug: term.slug,
          title: term.title,
          sectionId: term.sectionId,
          score: score,
          preview: this.generatePreview(term.content, searchLower)
        });
      }
    }

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);

    return results.slice(0, 20); // Return top 20 results
  }

  // Generate a preview snippet around the search query
  generatePreview(content, query) {
    const index = content.toLowerCase().indexOf(query);
    if (index === -1) {
      return content.substring(0, 150) + '...';
    }

    const start = Math.max(0, index - 50);
    const end = Math.min(content.length, index + query.length + 100);
    let preview = content.substring(start, end);

    if (start > 0) preview = '...' + preview;
    if (end < content.length) preview = preview + '...';

    return preview;
  }

  // Get GPU specifications
  getSpecs() {
    return this.specs;
  }

  // Slugify helper
  slugify(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-');
  }
}

// Create singleton instance
const glossary = new GlossaryLoader();

module.exports = glossary;
