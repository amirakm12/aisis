#!/bin/bash

# AISIS Creative Studio - Quick Start Script
# This script gets you up and running in minutes!

set -e

echo "🚀 AISIS Creative Studio v2.0.0 - Quick Start"
echo "=============================================="
echo ""

# Check if we're on a supported system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✓ Linux system detected"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✓ macOS system detected"
else
    echo "❌ Unsupported operating system: $OSTYPE"
    echo "This script supports Linux and macOS only."
    exit 1
fi

# Check for required tools
echo "Checking system requirements..."

if ! command -v git &> /dev/null; then
    echo "❌ Git is required but not installed"
    exit 1
fi
echo "✓ Git found"

if ! command -v cmake &> /dev/null; then
    echo "❌ CMake is required but not installed"
    echo "Install with: sudo apt-get install cmake (Ubuntu) or brew install cmake (macOS)"
    exit 1
fi
echo "✓ CMake found"

if ! command -v g++ &> /dev/null; then
    echo "❌ C++ compiler is required but not installed"
    echo "Install with: sudo apt-get install build-essential (Ubuntu) or xcode-select --install (macOS)"
    exit 1
fi
echo "✓ C++ compiler found"

echo ""
echo "🏗️  Building AISIS Creative Studio with performance optimizations..."
echo "This may take 5-10 minutes depending on your system."
echo ""

# Build the project
./build.sh Release $(nproc 2>/dev/null || sysctl -n hw.ncpu) true true true

echo ""
echo "🎉 AISIS Creative Studio is ready!"
echo ""
echo "🚀 To start the application:"
echo "   cd build && ./aisis_studio"
echo ""
echo "⚡ Performance profiles available:"
echo "   ./aisis_studio --profile high_performance    # Maximum performance"
echo "   ./aisis_studio --profile balanced           # Balanced performance"
echo "   ./aisis_studio --profile power_saving       # Battery optimized"
echo ""
echo "📊 To run benchmarks:"
echo "   cd build && ./tests/aisis_benchmarks"
echo ""
echo "📖 For more information, see README.md"
echo ""
echo "Enjoy creating with AISIS Creative Studio! 🎨🎵🤖"