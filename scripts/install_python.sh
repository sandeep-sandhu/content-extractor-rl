#!/bin/bash
# Install Python package

set -e

cd crates/article-extractor-py

echo "Installing article-extractor Python package..."

# Build and install in development mode
maturin develop --release

echo "Installation complete!"
echo ""
echo "Test with:"
echo "  python -c 'from article_extractor import RustArticleExtractor; print(\"OK\")'"
