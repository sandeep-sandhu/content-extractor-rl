# Article Extractor - RL-based HTML Article Extraction

A high-performance article extractor that uses Reinforcement Learning (DQN) to extract clean article content from HTML pages. Built in Rust with Python bindings.

## Features

-  **Fast**: Written in Rust for maximum performance
-  **Smart**: Uses Deep Q-Network (DQN) with curriculum learning
-  **Python-friendly**: Easy-to-use Python bindings via PyO3
-  **Adaptive**: Learns from site-specific patterns
-  **Accurate**: Multi-component reward function for quality extraction
-  **Configurable**: Environment variables for all settings

## Architecture
```
┌─────────────────────────────────────────┐
│           Python Interface              │
│  (PyO3 bindings + Maturin packaging)    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Rust Core Library               │
│  • BaselineExtractor (heuristics)       │
│  • DQN Agent (RL extraction)            │
│  • Environment & Replay Buffer          │
│  • Site Profile Memory                  │
│  • Training & Hyperparameter Search     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Model Inference                 │
│  • Candle (ML framework)                │
│  • ONNX model support                   │
└─────────────────────────────────────────┘
```

## **Project Structure**
````
article-extractor/
├── Cargo.toml (workspace)
├── scripts/
│   ├── build_all.sh
│   ├── install_python.sh
│   └── test_all.sh
├── examples/
│   ├── extract_example.py
│   └── complete_workflow.py
├── crates/
│   ├── article-extractor/ (core library)
│   ├── article-extractor-cli/ (CLI)
│   └── article-extractor-py/ (Python bindings)
├── README.md
└── USAGE.md
````

## **To Build and Use**:
````bash
# 1. Build everything
./scripts/build_all.sh

# 2. Install Python package
./scripts/install_python.sh

# 3. Test installation
python -c "from article_extractor import RustArticleExtractor; print('✓ Success!')"

# 4. Run example
python examples/complete_workflow.py

# 5. Use CLI
./target/release/article-extractor train --data-dir ./data --episodes 1000
````


# Installation

## From Python (PyPI)

```bash
pip install article-extractor
```

## From Source

```bash
# Clone repository
git clone https://github.com/yourusername/article-extractor
cd article-extractor

# Build Rust library
cargo build --release

# Build Python package
cd crates/article-extractor-py
maturin develop --release

# Or build wheel
maturin build --release
pip install target/wheels/article_extractor-*.whl
```

# Usage
## Python API

```python
from article_extractor import RustArticleExtractor

# Initialize extractor
extractor = RustArticleExtractor(
    model="path/to/model.onnx",  # Optional: trained model
    site_profile="path/to/profile.json"  # Optional: site profile
)

# Extract article
result = extractor.extract(
    website_page_html="<html>...</html>",
    url="https://example.com/article"
)

print(result['content'])
print(result['quality_score'])
```

## CLI Usage

```bash
# Extract article
article-extractor extract \
    --html-file article.html \
    --url https://example.com/article \
    --model model.onnx \
    --output result.json

# Train model
article-extractor train \
    --data-dir ./training_data \
    --episodes 10000 \
    --improved

# Hyperparameter search
article-extractor search \
    --data-dir ./training_data \
    --episodes-per-trial 500
```

## Environment Variables

```bash
# Model and data paths
export ARTICLE_EXTRACTOR_MODEL_PATH=/path/to/model.onnx
export ARTICLE_EXTRACTOR_SITE_PROFILES=/path/to/profiles
export ARTICLE_EXTRACTOR_OUTPUT_DIR=/path/to/output
```

## Training

### Prepare Training Data

Organize HTML files in a directory structure:
```
training_data/
├── example_com_article1.html
├── example_com_article2.html
├── news_site_article1.html
└── ...
```

## Train Model

```python
from article_extractor import RustArticleExtractor

# Load HTML samples
html_samples = [
    ("<html>...</html>", "https://example.com/article1"),
    ("<html>...</html>", "https://example.com/article2"),
]

# Initialize and train
extractor = RustArticleExtractor()
metrics = extractor.train(
    html_samples=html_samples,
    episodes=10000,
    improved=True  # Use curriculum learning
)

print(f"Best quality: {metrics['best_avg_quality']}")
```

# API Reference

## RustArticleExtractor

`__init__(site_profile=None, model=None)`

Initialize the extractor.
Parameters:

site_profile (str, optional): Path to site profile JSON
model (str, optional): Path to ONNX model file

`extract(website_page_html, url)`
Extract article from HTML content.
Parameters:

website_page_html (str): HTML content as string
url (str): URL of the page

Returns:

dict with keys: content, quality_score, url, title, date, method, xpath

`extract_batch(html_url_pairs)`
Extract multiple articles in batch.
Parameters:

html_url_pairs (list): List of (html, url) tuples

Returns:

dict with articles key containing list of results

`train(html_samples, episodes=1000, improved=False)`
Train the model.
Parameters:

html_samples (list): List of (html, url) tuples
episodes (int): Number of training episodes
improved (bool): Use improved training features

Returns:

dict with training metrics


# Development
## Building

```bash
# Build Rust library
cargo build --release

# Build Python package
cd crates/article-extractor-py
maturin develop

# Run tests
cargo test
```

### Project Structure
```
article-extractor/
├── Cargo.toml                 # Workspace configuration
├── crates/
│   ├── article-extractor/     # Core Rust library
│   ├── article-extractor-cli/ # CLI binary
│   └── article-extractor-py/  # Python bindings
└── README.md
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
```

---

## Part 18: Build Scripts and Final Setup

**`crates/article-extractor-py/build.rs`:**
```rust
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
```

**`.github/workflows/ci.yml`** (Optional CI/CD):
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
    - name: Run tests
      run: cargo test --all

  build-python:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install maturin
      run: pip install maturin
    
    - name: Build wheels
      run: |
        cd crates/article-extractor-py
        maturin build --release
    
    - name: Upload wheels
      uses: actions/upload-artifact@v3
      with:
        name: wheels
        path: crates/article-extractor-py/target/wheels/
```

**Example usage script `examples/extract_example.py`:**
```python
#!/usr/bin/env python3
"""
Example: Extract article from HTML file
"""

from article_extractor import ArticleExtractor
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_example.py <html_file> <url>")
        sys.exit(1)
    
    html_file = sys.argv[1]
    url = sys.argv[2]
    
    # Read HTML
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Initialize extractor
    extractor = ArticleExtractor()
    
    # Extract
    result = extractor.extract_from_html(html_content, url)
    
    # Print results
    print(f"URL: {result['url']}")
    print(f"Quality Score: {result['quality_score']:.3f}")
    print(f"Method: {result['method']}")
    print(f"\nExtracted Content ({len(result['content'])} chars):")
    print("=" * 80)
    print(result['content'][:500])
    if len(result['content']) > 500:
        print(f"\n... ({len(result['content']) - 500} more characters)")

if __name__ == '__main__':
    main()
```

---


##  **Build and test**:

```bash
# Build everything
cargo build --release

# Build Python package
cd crates/article-extractor-py
maturin develop --release

# Test in Python
python3 -c "from article_extractor import RustArticleExtractor; print('Success!')"
```
