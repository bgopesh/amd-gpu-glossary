#!/bin/bash

echo "============================================"
echo "AMD GPU Glossary - Starting Local Server"
echo "============================================"
echo ""

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "Starting server with Python on port 8000..."
    echo ""
    echo "Open your browser to: http://localhost:8000"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    python3 -m http.server 8000
elif command -v python &> /dev/null; then
    echo "Starting server with Python on port 8000..."
    echo ""
    echo "Open your browser to: http://localhost:8000"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    python -m http.server 8000
elif command -v node &> /dev/null; then
    echo "Starting server with Node.js on port 8000..."
    echo ""
    echo "Installing http-server if needed..."
    npx http-server -p 8000 -o
else
    echo "ERROR: Neither Python nor Node.js was found."
    echo ""
    echo "Please install one of the following:"
    echo "  - Python: https://www.python.org/downloads/"
    echo "  - Node.js: https://nodejs.org/"
    echo ""
    read -p "Press Enter to exit..."
fi
