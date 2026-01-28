// AMD GPU Glossary Application

class GlossaryApp {
    constructor() {
        this.terms = [];
        this.gpuSpecs = null;
        this.currentCategory = 'all';
        this.searchQuery = '';
        this.init();
    }

    async init() {
        await this.loadContent();
        this.setupEventListeners();
        this.render();
    }

    async loadContent() {
        try {
            // Load markdown files
            const files = [
                { file: 'gpu-glossary/01-device-hardware.md', category: 'device-hardware' },
                { file: 'gpu-glossary/02-device-software.md', category: 'device-software' },
                { file: 'gpu-glossary/03-host-software.md', category: 'host-software' },
                { file: 'gpu-glossary/04-performance.md', category: 'performance' }
            ];

            for (const { file, category } of files) {
                const content = await this.fetchFile(file);
                const terms = this.parseMarkdown(content, category);
                this.terms.push(...terms);
            }

            // Load GPU specs
            const specsContent = await this.fetchFile('gpu-glossary/amd-gpu-specs.json');
            this.gpuSpecs = JSON.parse(specsContent);

        } catch (error) {
            console.error('Error loading content:', error);
            this.showError('Failed to load glossary content');
        }
    }

    async fetchFile(path) {
        const response = await fetch(path);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${path}`);
        }
        return await response.text();
    }

    parseMarkdown(content, category) {
        const terms = [];
        const lines = content.split('\n');
        let currentTerm = null;
        let currentContent = [];
        let inCodeBlock = false;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            // Toggle code block state
            if (line.startsWith('```')) {
                inCodeBlock = !inCodeBlock;
                if (currentTerm) {
                    currentContent.push(line);
                }
                continue;
            }

            // Skip the main category header (# Device Hardware, etc.)
            if (line.startsWith('# ') && !currentTerm) {
                continue;
            }

            // Detect term header (## Term Name)
            if (line.startsWith('## ') && !inCodeBlock) {
                // Save previous term
                if (currentTerm) {
                    currentTerm.content = currentContent.join('\n').trim();
                    currentTerm.preview = this.generatePreview(currentTerm.content);
                    terms.push(currentTerm);
                }

                // Start new term
                const title = line.substring(3).trim();
                currentTerm = {
                    id: this.slugify(title),
                    title: title,
                    category: category,
                    content: '',
                    preview: ''
                };
                currentContent = [];
            } else if (currentTerm) {
                currentContent.push(line);
            }
        }

        // Save last term
        if (currentTerm) {
            currentTerm.content = currentContent.join('\n').trim();
            currentTerm.preview = this.generatePreview(currentTerm.content);
            terms.push(currentTerm);
        }

        return terms;
    }

    generatePreview(content) {
        // Remove code blocks (including ASCII diagrams)
        let preview = content.replace(/```[\s\S]*?```/g, '');

        // Remove markdown syntax for preview
        preview = preview
            .replace(/#{1,6}\s+/g, '') // Remove headers
            .replace(/\*\*/g, '') // Remove bold
            .replace(/\*/g, '') // Remove italic
            .replace(/`[^`]+`/g, '') // Remove inline code
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove links, keep text
            .replace(/^[-*+]\s+/gm, '') // Remove list markers
            .replace(/^\d+\.\s+/gm, '') // Remove numbered lists
            .trim();

        // Get first meaningful paragraph
        const lines = preview.split('\n').filter(line => line.trim().length > 0);
        const firstPara = lines.slice(0, 3).join(' ').trim();
        return firstPara.length > 200 ? firstPara.substring(0, 200) + '...' : firstPara;
    }

    slugify(text) {
        return text
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')
            .replace(/\s+/g, '-');
    }

    setupEventListeners() {
        // Search input
        const searchInput = document.getElementById('searchInput');
        searchInput.addEventListener('input', (e) => {
            this.searchQuery = e.target.value.toLowerCase();
            this.render();
        });

        // Category tabs
        const tabs = document.querySelectorAll('.tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                tabs.forEach(t => t.classList.remove('active'));
                e.target.classList.add('active');
                this.currentCategory = e.target.dataset.category;
                this.render();
            });
        });

        // Clear filter
        const clearFilter = document.getElementById('clearFilter');
        clearFilter.addEventListener('click', () => {
            this.searchQuery = '';
            searchInput.value = '';
            this.render();
        });

        // Modal close
        const modal = document.getElementById('termModal');
        const modalClose = document.getElementById('modalClose');

        modalClose.addEventListener('click', () => {
            modal.classList.remove('show');
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.classList.remove('show');
            }
        });

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.classList.contains('show')) {
                modal.classList.remove('show');
            }
        });
    }

    render() {
        const glossaryContent = document.getElementById('glossaryContent');
        const gpuSpecsContent = document.getElementById('gpuSpecsContent');
        const filterInfo = document.getElementById('filterInfo');
        const filterText = document.getElementById('filterText');

        // Show/hide sections based on category
        if (this.currentCategory === 'gpu-specs') {
            glossaryContent.style.display = 'none';
            gpuSpecsContent.style.display = 'block';
            this.renderGPUSpecs();
            return;
        } else {
            glossaryContent.style.display = 'grid';
            gpuSpecsContent.style.display = 'none';
        }

        // Filter terms
        let filteredTerms = this.terms;

        if (this.currentCategory !== 'all') {
            filteredTerms = filteredTerms.filter(term => term.category === this.currentCategory);
        }

        if (this.searchQuery) {
            filteredTerms = filteredTerms.filter(term =>
                term.title.toLowerCase().includes(this.searchQuery) ||
                term.content.toLowerCase().includes(this.searchQuery) ||
                term.category.toLowerCase().includes(this.searchQuery)
            );
        }

        // Show/hide filter info
        if (this.searchQuery) {
            filterInfo.style.display = 'flex';
            filterText.textContent = `Found ${filteredTerms.length} result${filteredTerms.length !== 1 ? 's' : ''} for "${this.searchQuery}"`;
        } else {
            filterInfo.style.display = 'none';
        }

        // Render terms
        if (filteredTerms.length === 0) {
            glossaryContent.innerHTML = this.renderNoResults();
        } else {
            glossaryContent.innerHTML = filteredTerms
                .map(term => this.renderTermCard(term))
                .join('');

            // Add click handlers
            document.querySelectorAll('.term-card').forEach(card => {
                card.addEventListener('click', () => {
                    const termId = card.dataset.termId;
                    this.showTermModal(termId);
                });
            });
        }
    }

    renderTermCard(term) {
        const categoryLabel = this.getCategoryLabel(term.category);

        return `
            <div class="term-card" data-term-id="${term.id}">
                <div class="term-title">
                    ${this.escapeHtml(term.title)}
                </div>
                <div class="term-category">${categoryLabel}</div>
                <div class="term-preview">
                    ${this.escapeHtml(term.preview)}
                </div>
            </div>
        `;
    }

    renderNoResults() {
        return `
            <div class="no-results">
                <div class="no-results-icon">🔍</div>
                <div class="no-results-text">No terms found</div>
                <p>Try adjusting your search or filters</p>
            </div>
        `;
    }

    showTermModal(termId) {
        const term = this.terms.find(t => t.id === termId);
        if (!term) return;

        const modal = document.getElementById('termModal');
        const modalBody = document.getElementById('modalBody');

        const categoryLabel = this.getCategoryLabel(term.category);
        const contentHtml = this.markdownToHtml(term.content);

        modalBody.innerHTML = `
            <div class="term-category">${categoryLabel}</div>
            <h2 style="margin: 16px 0 24px 0; font-size: 2rem;">${this.escapeHtml(term.title)}</h2>
            <div class="term-content">
                ${contentHtml}
            </div>
        `;

        modal.classList.add('show');
    }

    renderGPUSpecs() {
        const container = document.getElementById('gpuSpecsContent');
        if (!this.gpuSpecs) {
            container.innerHTML = '<div class="loading">Loading GPU specifications...</div>';
            return;
        }

        const gpuCards = this.gpuSpecs.amd_instinct_gpus.map(gpu => this.renderGPUCard(gpu)).join('');

        container.innerHTML = `
            <h2 style="margin-bottom: 24px; font-size: 2rem;">AMD Instinct GPU Specifications</h2>
            <div class="gpu-cards-grid">
                ${gpuCards}
            </div>
        `;
    }

    renderGPUCard(gpu) {
        return `
            <div class="gpu-card">
                <div class="gpu-card-header">
                    <div class="gpu-model">${gpu.model}</div>
                    <div class="gpu-arch">${gpu.architecture}</div>
                </div>
                <div class="gpu-specs">
                    ${this.renderSpec('Compute Units', gpu.compute_units)}
                    ${this.renderSpec('Wavefront Size', gpu.wavefront_size)}
                    ${this.renderSpec('Clock', `${gpu.base_clock_ghz} GHz`)}
                    ${this.renderSpec('Memory', `${gpu.memory_size_gb} GB ${gpu.memory_type}`)}
                    ${this.renderSpec('Bandwidth', `${gpu.memory_bandwidth_gbps} GB/s`)}
                    ${this.renderSpec('L3 Cache', `${gpu.l3_cache_mb} MB`)}
                    ${this.renderSpec('FP64', `${gpu.fp64_tflops} TFLOPS`)}
                    ${this.renderSpec('FP32', `${gpu.fp32_tflops} TFLOPS`)}
                    ${gpu.fp16_tflops ? this.renderSpec('FP16', `${gpu.fp16_tflops} TFLOPS`) : ''}
                    ${gpu.fp8_tflops ? this.renderSpec('FP8', `${gpu.fp8_tflops} TFLOPS`) : ''}
                    ${this.renderSpec('TDP', `${gpu.tdp_watts}W`)}
                    ${this.renderSpec('Process', gpu.process_node)}
                </div>
            </div>
        `;
    }

    renderSpec(label, value) {
        return `
            <div class="gpu-spec-row">
                <span class="spec-label">${label}</span>
                <span class="spec-value">${value}</span>
            </div>
        `;
    }

    getCategoryLabel(category) {
        const labels = {
            'device-hardware': 'Device Hardware',
            'device-software': 'Device Software',
            'host-software': 'Host Software',
            'performance': 'Performance'
        };
        return labels[category] || category;
    }

    markdownToHtml(markdown) {
        let html = markdown;

        // Code blocks (preserve whitespace for ASCII diagrams)
        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            // Don't trim ASCII diagrams - preserve exact spacing
            return `<pre><code>${this.escapeHtml(code).replace(/^\n+/, '').replace(/\n+$/, '')}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Headers (###, ####, etc.)
        html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');

        // Lists
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // Links (internal references like [Term](#term))
        html = html.replace(/\[([^\]]+)\]\(#[^)]+\)/g, '<strong>$1</strong>');

        // Regular links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

        // Paragraphs
        html = html.split('\n\n').map(para => {
            para = para.trim();
            if (!para) return '';
            if (para.startsWith('<')) return para;
            return `<p>${para.replace(/\n/g, '<br>')}</p>`;
        }).join('\n');

        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showError(message) {
        const glossaryContent = document.getElementById('glossaryContent');
        glossaryContent.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">⚠️</div>
                <div class="no-results-text">Error</div>
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new GlossaryApp();
});
