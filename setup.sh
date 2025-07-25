#!/bin/bash

# AISIS Creative Studio Environment Setup Script
# This script sets up the development environment for the project

echo "Setting up AISIS Creative Studio development environment..."

# Add vcpkg to PATH
export PATH="/tmp/vcpkg:$PATH"

# Verify vcpkg is available
if command -v vcpkg &> /dev/null; then
    echo "✓ vcpkg is available"
    vcpkg --version
else
    echo "✗ vcpkg is not available"
    exit 1
fi

# Verify build tools
echo "Checking build tools..."
for tool in gcc g++ cmake make; do
    if command -v $tool &> /dev/null; then
        echo "✓ $tool is available"
    else
        echo "✗ $tool is not available"
    fi
done

# Create build directory structure if it doesn't exist
mkdir -p aisis/build/Debug

echo "Environment setup complete!"
echo "To use vcpkg in your session, run: export PATH=\"/tmp/vcpkg:\$PATH\""