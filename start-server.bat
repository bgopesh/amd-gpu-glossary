@echo off
echo ============================================
echo AMD GPU Glossary - Starting Local Server
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo Starting server with Python on port 8000...
    echo.
    echo Open your browser to: http://localhost:8000
    echo.
    echo Press Ctrl+C to stop the server
    echo.
    python -m http.server 8000
) else (
    REM Check if Node.js is available
    node --version >nul 2>&1
    if %errorlevel% == 0 (
        echo Starting server with Node.js on port 8000...
        echo.
        echo Installing http-server if needed...
        call npx http-server -p 8000 -o
    ) else (
        echo ERROR: Neither Python nor Node.js was found.
        echo.
        echo Please install one of the following:
        echo   - Python: https://www.python.org/downloads/
        echo   - Node.js: https://nodejs.org/
        echo.
        pause
    )
)
