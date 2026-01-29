// AMD GPU Glossary Frontend Application

class GlossaryApp {
  constructor() {
    this.sections = [];
    this.currentTerm = null;
    this.searchTimeout = null;
    this.currentTheme = 'amd-red';

    // DOM elements
    this.navSections = document.getElementById('nav-sections');
    this.contentArea = document.getElementById('content-area');
    this.searchInput = document.getElementById('search-input');
    this.searchResults = document.getElementById('search-results');
    this.loading = document.getElementById('loading');
    this.themeSelect = document.getElementById('theme-select');

    // Initialize
    this.init();
  }

  async init() {
    // Initialize theme
    this.initializeTheme();

    // Load sections
    await this.loadSections();

    // Set up event listeners
    this.setupEventListeners();

    // Handle initial route
    this.handleRoute();

    // Listen for hash changes
    window.addEventListener('hashchange', () => this.handleRoute());
  }

  // Initialize theme from localStorage or default
  initializeTheme() {
    const savedTheme = localStorage.getItem('glossary-theme') || 'amd-red';
    this.setTheme(savedTheme);
    if (this.themeSelect) {
      this.themeSelect.value = savedTheme;
    }
  }

  // Set theme
  setTheme(theme) {
    this.currentTheme = theme;
    document.body.setAttribute('data-theme', theme);
    localStorage.setItem('glossary-theme', theme);
  }

  // Load sections from API
  async loadSections() {
    try {
      this.showLoading();
      const response = await fetch('/api/sections');
      if (!response.ok) throw new Error('Failed to load sections');

      this.sections = await response.json();
      this.renderNavigation();
    } catch (error) {
      console.error('Error loading sections:', error);
      this.showError('Failed to load glossary sections');
    } finally {
      this.hideLoading();
    }
  }

  // Render navigation sidebar
  renderNavigation() {
    this.navSections.innerHTML = '';

    for (const section of this.sections) {
      const sectionEl = document.createElement('div');
      sectionEl.className = 'nav-section';
      sectionEl.dataset.sectionId = section.id;

      // Section title
      const titleEl = document.createElement('div');
      titleEl.className = 'nav-section-title';
      titleEl.innerHTML = `
        <span>${section.title}</span>
        <span class="nav-section-toggle">▼</span>
      `;
      titleEl.addEventListener('click', () => this.toggleSection(section.id));

      // Section terms
      const termsEl = document.createElement('div');
      termsEl.className = 'nav-section-terms';

      for (const term of section.terms) {
        const termLink = document.createElement('a');
        termLink.href = `#${term.id}`;
        termLink.className = 'nav-term';
        termLink.textContent = term.title;
        termLink.dataset.termId = term.id;
        termsEl.appendChild(termLink);
      }

      sectionEl.appendChild(titleEl);
      sectionEl.appendChild(termsEl);
      this.navSections.appendChild(sectionEl);
    }
  }

  // Toggle section collapse
  toggleSection(sectionId) {
    const sectionEl = this.navSections.querySelector(`[data-section-id="${sectionId}"]`);
    if (sectionEl) {
      sectionEl.classList.toggle('collapsed');
    }
  }

  // Set up event listeners
  setupEventListeners() {
    // Theme selector
    if (this.themeSelect) {
      this.themeSelect.addEventListener('change', (e) => {
        this.setTheme(e.target.value);
      });
    }

    // Search input
    this.searchInput.addEventListener('input', (e) => {
      clearTimeout(this.searchTimeout);
      const query = e.target.value.trim();

      if (query.length === 0) {
        this.hideSearchResults();
        return;
      }

      // Debounce search
      this.searchTimeout = setTimeout(() => {
        this.performSearch(query);
      }, 300);
    });

    // Close search results when clicking outside
    document.addEventListener('click', (e) => {
      if (!this.searchInput.contains(e.target) && !this.searchResults.contains(e.target)) {
        this.hideSearchResults();
      }
    });

    // Handle clicks on term links in content (for related terms)
    this.contentArea.addEventListener('click', (e) => {
      if (e.target.tagName === 'A' && e.target.hash) {
        e.preventDefault();
        const hash = e.target.hash.substring(1);
        window.location.hash = hash;
      }
    });
  }

  // Perform search
  async performSearch(query) {
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error('Search failed');

      const results = await response.json();
      this.renderSearchResults(results);
    } catch (error) {
      console.error('Search error:', error);
    }
  }

  // Render search results
  renderSearchResults(results) {
    if (results.length === 0) {
      this.searchResults.innerHTML = '<div style="padding: 1rem; color: var(--text-muted);">No results found</div>';
      this.searchResults.classList.remove('hidden');
      return;
    }

    this.searchResults.innerHTML = '';

    for (const result of results) {
      const resultEl = document.createElement('div');
      resultEl.className = 'search-result-item';
      resultEl.innerHTML = `
        <div class="search-result-title">${this.escapeHtml(result.title)}</div>
        <div class="search-result-section">${this.getSectionTitle(result.sectionId)}</div>
        <div class="search-result-preview">${this.escapeHtml(result.preview)}</div>
      `;

      resultEl.addEventListener('click', () => {
        window.location.hash = result.id;
        this.hideSearchResults();
        this.searchInput.value = '';
      });

      this.searchResults.appendChild(resultEl);
    }

    this.searchResults.classList.remove('hidden');
  }

  // Hide search results
  hideSearchResults() {
    this.searchResults.classList.add('hidden');
  }

  // Get section title by ID
  getSectionTitle(sectionId) {
    const section = this.sections.find(s => s.id === sectionId);
    return section ? section.title : sectionId;
  }

  // Handle route changes
  async handleRoute() {
    const hash = window.location.hash.substring(1);

    if (!hash || hash === '') {
      this.showWelcome();
      this.updateActiveNavItem(null);
      return;
    }

    if (hash === 'specs') {
      await this.showSpecs();
      this.updateActiveNavItem(null);
      return;
    }

    // Load term
    await this.loadTerm(hash);
  }

  // Show welcome screen
  showWelcome() {
    this.contentArea.innerHTML = `
      <div class="welcome">
        <h2>Welcome to the AMD GPU Glossary</h2>
        <p>Select a term from the navigation or use the search to get started.</p>

        <div class="welcome-sections">
          <div class="welcome-section">
            <h3>Device Hardware</h3>
            <p>Physical components and architecture of AMD GPUs</p>
          </div>
          <div class="welcome-section">
            <h3>Device Software</h3>
            <p>Software running on GPU (kernels, ISA)</p>
          </div>
          <div class="welcome-section">
            <h3>Host Software</h3>
            <p>CPU-side software and APIs (HIP, ROCm)</p>
          </div>
          <div class="welcome-section">
            <h3>Performance</h3>
            <p>Optimization and profiling tools</p>
          </div>
        </div>
      </div>
    `;
  }

  // Load and display a term
  async loadTerm(termId) {
    try {
      this.showLoading();

      let apiUrl;
      const parts = termId.split('/');

      if (parts.length === 2) {
        // Full format: section-id/term-slug
        const [sectionId, slug] = parts;
        apiUrl = `/api/terms/${sectionId}/${slug}`;
      } else if (parts.length === 1) {
        // Just slug - let backend search across sections
        apiUrl = `/api/terms/${termId}`;
      } else {
        this.showError('Invalid term ID');
        return;
      }

      const response = await fetch(apiUrl);
      if (!response.ok) {
        if (response.status === 404) {
          this.showError('Term not found');
        } else {
          throw new Error('Failed to load term');
        }
        return;
      }

      const term = await response.json();
      this.renderTerm(term);
      this.updateActiveNavItem(term.id);

      // Scroll to top
      this.contentArea.parentElement.scrollTop = 0;
    } catch (error) {
      console.error('Error loading term:', error);
      this.showError('Failed to load term');
    } finally {
      this.hideLoading();
    }
  }

  // Render term content
  renderTerm(term) {
    const sectionTitle = this.getSectionTitle(term.sectionId);

    this.contentArea.innerHTML = `
      <div class="term-content">
        <div class="term-header">
          <div class="term-section">${this.escapeHtml(sectionTitle)}</div>
          <h1 class="term-title">${this.escapeHtml(term.title)}</h1>
        </div>
        <div class="term-body">
          ${term.html}
        </div>
      </div>
    `;

    this.currentTerm = term;
  }

  // Show GPU specifications
  async showSpecs() {
    try {
      this.showLoading();

      const response = await fetch('/api/specs');
      if (!response.ok) throw new Error('Failed to load specs');

      const specs = await response.json();
      this.renderSpecs(specs);
    } catch (error) {
      console.error('Error loading specs:', error);
      this.showError('Failed to load GPU specifications');
    } finally {
      this.hideLoading();
    }
  }

  // Render GPU specifications
  renderSpecs(specs) {
    let html = `
      <div class="specs-container">
        <h1 class="specs-title">AMD Instinct GPU Specifications</h1>
        <div class="specs-grid">
    `;

    for (const gpu of specs.amd_instinct_gpus) {
      html += `
        <div class="spec-card">
          <div class="spec-card-header">
            <h2 class="spec-card-title">${this.escapeHtml(gpu.model)}</h2>
            <div class="spec-card-subtitle">${this.escapeHtml(gpu.architecture)} (${this.escapeHtml(gpu.gfxip)})</div>
          </div>
          <div class="spec-card-body">
            ${this.renderSpecItem('Compute Units', gpu.compute_units)}
            ${this.renderSpecItem('Memory', `${gpu.memory_size_gb} GB ${gpu.memory_type}`)}
            ${this.renderSpecItem('Memory Bandwidth', `${gpu.memory_bandwidth_gbps} GB/s`)}
            ${this.renderSpecItem('Base Clock', `${gpu.base_clock_ghz} GHz`)}
            ${this.renderSpecItem('L3 Cache', `${gpu.l3_cache_mb} MB`)}
            ${this.renderSpecItem('FP64', `${gpu.fp64_tflops} TFLOPS`)}
            ${this.renderSpecItem('FP32', `${gpu.fp32_tflops} TFLOPS`)}
            ${this.renderSpecItem('FP16', `${gpu.fp16_tflops} TFLOPS`)}
            ${gpu.fp8_tflops ? this.renderSpecItem('FP8', `${gpu.fp8_tflops} TFLOPS`) : ''}
            ${this.renderSpecItem('TDP', `${gpu.tdp_watts} W`)}
            ${this.renderSpecItem('Release Year', gpu.release_year)}
          </div>
        </div>
      `;
    }

    html += `
        </div>
      </div>
    `;

    this.contentArea.innerHTML = html;
    this.contentArea.parentElement.scrollTop = 0;
  }

  // Render a single spec item
  renderSpecItem(label, value) {
    return `
      <div class="spec-item">
        <div class="spec-label">${this.escapeHtml(label)}</div>
        <div class="spec-value">${this.escapeHtml(String(value))}</div>
      </div>
    `;
  }

  // Update active navigation item
  updateActiveNavItem(termId) {
    // Remove all active classes
    const allTermLinks = this.navSections.querySelectorAll('.nav-term');
    allTermLinks.forEach(link => link.classList.remove('active'));

    if (termId) {
      // Add active class to current term
      const activeLink = this.navSections.querySelector(`[data-term-id="${termId}"]`);
      if (activeLink) {
        activeLink.classList.add('active');

        // Ensure parent section is expanded
        const parentSection = activeLink.closest('.nav-section');
        if (parentSection) {
          parentSection.classList.remove('collapsed');
        }

        // Scroll into view if needed
        activeLink.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }

  // Show error message
  showError(message) {
    this.contentArea.innerHTML = `
      <div class="welcome">
        <h2 style="color: var(--amd-red);">Error</h2>
        <p>${this.escapeHtml(message)}</p>
      </div>
    `;
  }

  // Show loading indicator
  showLoading() {
    this.loading.classList.remove('hidden');
  }

  // Hide loading indicator
  hideLoading() {
    this.loading.classList.add('hidden');
  }

  // Escape HTML to prevent XSS
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new GlossaryApp();
});
