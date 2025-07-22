#!/bin/bash
# AISIS One-Click Installer

set -e

REPO_URL="https://github.com/YOUR-USERNAME/aisis.git"
INSTALL_DIR="/workspace/aisis"
VENV_DIR="$INSTALL_DIR/venv"

echo "Starting AISIS installation..."

# Check prerequisites
command -v git >/dev/null 2>&1 || { echo "Git not installed"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python3 not installed"; exit 1; }

# Clone or pull
if [ ! -d "$INSTALL_DIR" ]; then
  git clone "$REPO_URL" "$INSTALL_DIR"
else
  cd "$INSTALL_DIR"
  git pull
fi

cd "$INSTALL_DIR"

# Venv
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install deps
pip install -e .
pip install -e .[dev]

# Setup
python scripts/setup_environment.py

# Download models
python scripts/download_models.py

echo "Installation complete!"
