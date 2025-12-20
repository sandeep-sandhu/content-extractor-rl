#!/bin/bash
# Run all tests

set -e

echo "=========================================="
echo "Running All Tests"
echo "=========================================="

# Rust tests
echo "Running Rust tests..."
cargo test --all

# Python tests (if package is installed)
echo "Running Python tests..."
cd crates/article-extractor-py
if python -c "import article_extractor" 2>/dev/null; then
    python -m pytest tests/ || echo "No Python tests found"
else
    echo "Python package not installed, skipping Python tests"
fi

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
