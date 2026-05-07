# Content Extractor RL Python Package

High-performance article extraction from HTML using Reinforcement Learning, powered by Rust.

## Building from Source

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python dependencies
pip install maturin
```

### Build and Install
```bash
# Development mode (for testing)
cd crates/content-extractor-rl-py
maturin develop --release

# Build wheel for distribution
maturin build --release

# Install the wheel
pip install target/wheels/content_extractor_rl-*.whl
```

### Build for Multiple Python Versions
```bash
# Build for Python 3.8, 3.9, 3.10, 3.11
maturin build --release --interpreter python3.8
maturin build --release --interpreter python3.9
maturin build --release --interpreter python3.10
maturin build --release --interpreter python3.11
```

## Usage Example
```python
from content_extractor_rl import RustArticleExtractor

# Initialize
extractor = RustArticleExtractor(
    model="path/to/model.onnx",
    site_profile="path/to/profile.json"
)

# Extract article
html = """
<html>
<body>
    <article>
        <h1>Article Title</h1>
        <p>Article content here...</p>
    </article>
</body>
</html>
"""

result = extractor.extract(
    website_page_html=html,
    url="https://example.com/article"
)

print(f"Content: {result['content']}")
print(f"Quality: {result['quality_score']}")
```

## Environment Variables
```bash
export ARTICLE_EXTRACTOR_MODEL_PATH=/path/to/model.onnx
export ARTICLE_EXTRACTOR_SITE_PROFILES=/path/to/profiles
export ARTICLE_EXTRACTOR_OUTPUT_DIR=/path/to/output
```

## Testing
```bash
# Run Rust tests
cargo test

# Test Python package
python -c "from content_extractor_rl import RustArticleExtractor; print('OK')"
```
