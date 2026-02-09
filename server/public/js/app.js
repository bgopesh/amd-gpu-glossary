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

    // Add Home section
    const homeSection = document.createElement('div');
    homeSection.className = 'nav-section home-section';
    homeSection.innerHTML = `
      <div class="nav-section-title">
        <a href="#home" class="nav-section-title-link">🏠 Home</a>
      </div>
    `;
    this.navSections.appendChild(homeSection);

    for (const section of this.sections) {
      const sectionEl = document.createElement('div');
      sectionEl.className = 'nav-section';
      sectionEl.dataset.sectionId = section.id;

      // Section title - make it clickable to show topic list
      const titleEl = document.createElement('div');
      titleEl.className = 'nav-section-title';
      titleEl.innerHTML = `
        <a href="#section/${section.id}" class="nav-section-title-link">${section.title}</a>
        <span class="nav-section-toggle">▼</span>
      `;
      titleEl.querySelector('.nav-section-toggle').addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        this.toggleSection(section.id);
      });

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
        // Check if this is an internal anchor link (starts with # and targets an element on the current page)
        const hash = e.target.hash.substring(1);
        const targetElement = document.getElementById(hash);

        // If the target element exists on the current page, just scroll to it
        if (targetElement) {
          e.preventDefault();
          targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
          return;
        }

        // Otherwise, navigate to the term/page
        e.preventDefault();
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

    if (!hash || hash === '' || hash === 'home') {
      this.showHome();
      this.updateActiveNavItem(null);
      return;
    }

    if (hash === 'specs') {
      await this.showSpecs();
      this.updateActiveNavItem(null);
      return;
    }

    // Check if it's a section view (list of topics)
    if (hash.startsWith('section/')) {
      const sectionId = hash.substring(8); // Remove 'section/' prefix
      this.showSectionTopics(sectionId);
      this.updateActiveNavItem(null);
      return;
    }

    // Check if it's an overview page
    if (hash.startsWith('overview/')) {
      const sectionId = hash.substring(9); // Remove 'overview/' prefix
      await this.showOverview(sectionId);
      this.updateActiveNavItem(null);
      return;
    }

    // Load term
    await this.loadTerm(hash);
  }

  // Show home page with AMD logo
  showHome() {
    this.contentArea.innerHTML = `
      <div class="welcome">
        <div class="welcome-hero">
          <div class="amd-logo-container">
            <img src="/images/amd-logo.svg" alt="AMD Logo" class="amd-official-logo" />
          </div>
          <div class="welcome-hero-content">
            <h1 class="welcome-title">AMD GPU Glossary</h1>
            <p class="welcome-subtitle">Your comprehensive reference for AMD Instinct GPU computing</p>
            <p class="welcome-description">
              Explore detailed documentation on AMD CDNA architecture, ROCm software stack,
              HIP programming, and performance optimization for high-performance computing.
            </p>
          </div>
        </div>

        <div class="welcome-stats">
          <div class="stat-card">
            <div class="stat-value">304</div>
            <div class="stat-label">Compute Units (MI300X)</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">192 GB</div>
            <div class="stat-label">HBM3 Memory</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">1.3 PF</div>
            <div class="stat-label">FP16 Performance</div>
          </div>
          <div class="stat-card">
            <div class="stat-value">5.3 TB/s</div>
            <div class="stat-label">Memory Bandwidth</div>
          </div>
        </div>

        <div class="welcome-sections">
          <a href="#section/device-hardware" class="welcome-section device-hardware">
            <div class="section-icon">🔧</div>
            <h3>Device Hardware</h3>
            <p>Physical components and architecture of AMD GPUs including compute units, memory hierarchy, and chiplet design</p>
            <div class="section-arrow">→</div>
          </a>
          <a href="#device-software/kernel" class="welcome-section device-software">
            <div class="section-icon">⚡</div>
            <h3>Device Software</h3>
            <p>GPU kernels, ISA, wavefronts, and low-level execution model for parallel computing</p>
            <div class="section-arrow">→</div>
          </a>
          <a href="#host-software/hip-heterogeneous-compute-interface-for-portability" class="welcome-section host-software">
            <div class="section-icon">💻</div>
            <h3>Host Software</h3>
            <p>HIP programming, ROCm platform, compilers, and CPU-side APIs for GPU development</p>
            <div class="section-arrow">→</div>
          </a>
          <a href="#performance/rocprof-rocm-profiler" class="welcome-section performance">
            <div class="section-icon">📊</div>
            <h3>Performance</h3>
            <p>Profiling tools, optimization techniques, and performance analysis for maximum throughput</p>
            <div class="section-arrow">→</div>
          </a>
        </div>

        <div class="welcome-footer">
          <p class="footer-note">
            💡 <strong>Getting Started:</strong> Use the search bar or navigate through sections on the left.
            Click any section to explore GPU computing topics.
          </p>
        </div>
      </div>
    `;
  }

  // Show section topics list
  showSectionTopics(sectionId) {
    const section = this.sections.find(s => s.id === sectionId);

    if (!section) {
      this.showError('Section not found');
      return;
    }

    this.contentArea.innerHTML = `
      <div class="section-topics">
        <div class="section-topics-header">
          <div class="section-icon-large">${this.getSectionIcon(sectionId)}</div>
          <div class="section-header-content">
            <h1 class="section-topics-title">${this.escapeHtml(section.title)}</h1>
            <p class="section-topics-description">${this.escapeHtml(section.description)}</p>
          </div>
        </div>

        <div class="topics-grid">
          ${section.terms.map(term => `
            <a href="#${term.id}" class="topic-card">
              <div class="topic-card-icon">📄</div>
              <div class="topic-card-content">
                <h3 class="topic-card-title">${this.escapeHtml(term.title)}</h3>
                <div class="topic-card-arrow">→</div>
              </div>
            </a>
          `).join('')}
        </div>
      </div>
    `;

    this.contentArea.parentElement.scrollTop = 0;
  }

  // Get section icon
  getSectionIcon(sectionId) {
    const icons = {
      'device-hardware': '🔧',
      'device-software': '⚡',
      'host-software': '💻',
      'performance': '📊'
    };
    return icons[sectionId] || '📁';
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

    // Render diagrams after content is loaded
    this.renderDiagrams();
  }

  // Render diagrams in the current content
  renderDiagrams() {
    if (typeof DiagramRenderer !== 'undefined') {
      setTimeout(() => {
        const renderer = new DiagramRenderer();
        renderer.renderDiagrams();
      }, 100);
    }
  }

  // Show section overview
  async showOverview(sectionId) {
    try {
      this.showLoading();

      const response = await fetch(`/api/overview/${sectionId}`);
      if (!response.ok) {
        if (response.status === 404) {
          this.showError('Overview not found');
        } else {
          throw new Error('Failed to load overview');
        }
        return;
      }

      const overview = await response.json();
      this.renderOverview(overview);
    } catch (error) {
      console.error('Error loading overview:', error);
      this.showError('Failed to load section overview');
    } finally {
      this.hideLoading();
    }
  }

  // Render section overview
  renderOverview(overview) {
    this.contentArea.innerHTML = `
      <div class="term-content">
        <div class="term-header">
          <div class="term-section">Overview</div>
          <h1 class="term-title">${this.escapeHtml(overview.title)}</h1>
        </div>
        <div class="term-body">
          ${overview.html}
        </div>
      </div>
    `;

    this.contentArea.parentElement.scrollTop = 0;

    // Render diagrams after content is loaded
    this.renderDiagrams();
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
