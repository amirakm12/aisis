#!/bin/bash

echo "========================================"
echo "AISIS Full Installation Test"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "Python found. Running installation test..."
echo

# Run the installation test
python3 scripts/installation_test.py

echo
echo "Test completed. Check the output above for results."
echo "Detailed results saved to test_results.json"
echo 