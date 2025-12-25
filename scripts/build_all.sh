#!/bin/bash
# Build all components of the article extractor

set -e

echo "=========================================="
echo "Building Article Extractor"
echo "=========================================="

echo "Building with CUDA support..."

# Check for CUDA
if command -v nvcc &> /dev/null; then
    echo "✅ CUDA compiler found: $(nvcc --version | head -n1)"
else
    echo "⚠️  nvcc not found. Install CUDA Toolkit from:"
    echo "    https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Build Rust library
echo "Building Rust core library..."
cargo build --release --features cuda,mlflow-rs

# Build CLI
echo "Building CLI binary..."
cargo build --release --bin article-extractor --features cuda,mlflow-rs

# Build Python package
echo "Building Python package..."
cd crates/article-extractor-py

# Build for current Python version
maturin develop --release --features cuda

# Optional: Build for multiple Python versions
if [ "$BUILD_ALL_PYTHON" = "1" ]; then
    echo "Building for multiple Python versions..."
    for version in 3.8 3.9 3.10 3.11; do
        if command -v python$version &> /dev/null; then
            echo "Building for Python $version..."
            maturin build --release --interpreter python$version --features cuda
        fi
    done
fi

cd ../..

echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Binaries:"
echo "  CLI: target/release/article-extractor"
echo "Python wheels:"
echo "  $(ls crates/article-extractor-py/target/wheels/*.whl 2>/dev/null || echo 'None')"

pip3 install -U crates/article-extractor-py/target/wheels/*.whl

