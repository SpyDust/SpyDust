#!/bin/bash

# Grain Size Distribution GUI Launcher
# This script launches the interactive GUI for exploring grain size parameters

echo "Starting Grain Size Distribution GUI..."
echo "Loading SpyDust modules and initializing emulators..."
echo "This may take a moment on first run."
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please ensure Python is installed and in your PATH."
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import PyQt5, matplotlib, numpy, scipy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required dependencies..."
    pip install -r gui_requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies. Please install manually:"
        echo "pip install PyQt5 matplotlib numpy scipy"
        exit 1
    fi
fi

# Launch the GUI
echo "Launching GUI..."
python grain_size_gui.py

echo "GUI closed."