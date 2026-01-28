# AMD GPU Glossary - Web Application

A beautiful, self-hosted web interface for the AMD GPU Glossary, inspired by Modal's GPU glossary design.

## Features

- **Searchable Interface**: Quickly find terms, definitions, and concepts
- **Category Filtering**: Browse by Device Hardware, Device Software, Host Software, or Performance
- **GPU Specifications**: Interactive cards displaying detailed specs for AMD Instinct GPUs (MI300X, MI250X, MI210, MI100, etc.)
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Clean UI**: Modern, AMD-branded interface with smooth animations
- **No Build Required**: Pure HTML/CSS/JavaScript - just serve and use

## Quick Start

### Option 1: Python (Recommended)

If you have Python installed:

```bash
# Navigate to the project directory
cd amd-gpu-glossary

# Start the server
python -m http.server 8000

# Or use npm script
npm start
```

Then open your browser to: http://localhost:8000

### Option 2: Node.js

If you have Node.js installed:

```bash
# Install http-server (one time only)
npm install

# Start the server
npm run start:node
```

This will automatically open http://localhost:8000 in your browser.

### Option 3: Windows Batch File

Double-click the `start-server.bat` file included in the directory.

### Option 4: Any Web Server

You can use any static web server:
- Live Server (VS Code extension)
- nginx
- Apache
- Caddy
- Any other HTTP server

## File Structure

```
amd-gpu-glossary/
├── index.html              # Main HTML structure
├── styles.css              # All styling and responsive design
├── app.js                  # JavaScript application logic
├── gpu-glossary/           # Content directory
│   ├── 01-device-hardware.md
│   ├── 02-device-software.md
│   ├── 03-host-software.md
│   ├── 04-performance.md
│   └── amd-gpu-specs.json
├── package.json            # Node.js configuration
└── WEB_README.md          # This file
```

## How It Works

1. **Content Loading**: The app dynamically loads markdown files from the `gpu-glossary/` directory
2. **Parsing**: Markdown is parsed into structured terms with titles, categories, and content
3. **Rendering**: Terms are displayed as interactive cards in a responsive grid
4. **Search**: Real-time search across term titles, content, and categories
5. **Modal View**: Click any term card to view the full definition in a modal

## Customization

### Styling

Edit `styles.css` to customize:
- Colors (see CSS variables in `:root`)
- Layout and spacing
- Typography
- Animations

### Content

Add or modify markdown files in the `gpu-glossary/` directory. The app automatically parses any terms defined with `## Term Name` headers.

### GPU Specs

Edit `gpu-glossary/amd-gpu-specs.json` to add or update GPU specifications.

## Browser Compatibility

Works on all modern browsers:
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

## Deployment

### GitHub Pages

1. Push your repository to GitHub
2. Go to Settings > Pages
3. Select your main branch as the source
4. Your site will be live at `https://yourusername.github.io/amd-gpu-glossary/`

### Netlify

1. Drag and drop the entire directory to Netlify
2. Or connect your GitHub repository
3. No build configuration needed

### Vercel

1. Install Vercel CLI: `npm i -g vercel`
2. Run: `vercel`
3. Follow the prompts

### Any Static Host

Upload all files to any static hosting service:
- AWS S3 + CloudFront
- Google Cloud Storage
- Azure Static Web Apps
- Firebase Hosting
- Cloudflare Pages

## Development

The application is built with vanilla JavaScript - no frameworks or build tools required.

### Adding New Categories

1. Create a new markdown file in `gpu-glossary/`
2. Update the `files` array in `app.js` to include your new file
3. Add a tab button in `index.html` with the appropriate `data-category` attribute
4. Update the `getCategoryLabel()` method in `app.js`

### Markdown Format

Terms should follow this structure:

```markdown
## Term Name

Brief description or key characteristics.

**Key characteristics:**
- Feature 1
- Feature 2

**Related:** [Other Term](#other-term)
```

## Performance

- Loads all content on initial page load
- Client-side search and filtering (no server required)
- Lazy rendering of modal content
- Optimized for thousands of terms

## License

- **Documentation Content**: Creative Commons Attribution 4.0 International License (CC BY 4.0)
- **Web Application Code**: MIT License

## Credits

Inspired by [Modal's GPU Glossary](https://modal.com/gpu-glossary) for NVIDIA GPUs.

Built for the AMD GPU and ROCm community.
