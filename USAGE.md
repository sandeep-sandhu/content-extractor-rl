# Article Extractor - Complete Usage Guide

## Installation

### Option 1: Install from PyPI (when published)
```bash
pip install article-extractor
```

### Option 2: Build from Source
```bash
# Clone repository
git clone https://github.com/yourusername/article-extractor
cd article-extractor

# Build and install
./scripts/install_python.sh
```

## Quick Start

### Python API
```python
from article_extractor import RustArticleExtractor

# Initialize with trained model
extractor = RustArticleExtractor(
    model="path/to/model.onnx",
    site_profile="path/to/profile.json"  # Optional
)

# Extract article
result = extractor.extract(
    website_page_html="<html>...</html>",
    url="https://example.com/article"
)

print(result['content'])
```

### Command Line
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

## Training

### Prepare Training Data

Create directory with HTML files:
```
training_data/
├── article1.html
├── article2.html
└── ...
```

### Train with Python

```python
from article_extractor import RustArticleExtractor

# Load samples
html_samples = [
    ("<html>...</html>", "https://example.com/1"),
    ("<html>...</html>", "https://example.com/2"),
]

# Train
extractor = RustArticleExtractor()
metrics = extractor.train(
    html_samples=html_samples,
    episodes=10000,
    improved=True
)

print(f"Best quality: {metrics['best_avg_quality']}")
```

## Advanced Features
### Batch Extraction

```python
html_url_pairs = [
    ("<html>...</html>", "https://site1.com/article"),
    ("<html>...</html>", "https://site2.com/article"),
]

results = extractor.extract_batch(html_url_pairs)

for article in results['articles']:
    print(f"{article['url']}: {article['quality_score']}")
```

## Resume from Checkpoint

```python
# Training automatically saves checkpoints
# Resume by running train again - it will detect and load the latest checkpoint
metrics = extractor.train(html_samples, episodes=20000, improved=True)
```

# Environment Variables

```bash
# Model path
export ARTICLE_EXTRACTOR_MODEL_PATH=/path/to/model.onnx

# Site profiles directory
export ARTICLE_EXTRACTOR_SITE_PROFILES=/path/to/profiles

# Output directory
export ARTICLE_EXTRACTOR_OUTPUT_DIR=/path/to/output
```

# Performance Tips

  - Use improved training for better quality
  - Enable ONNX feature for better model serialization:

```bash
cargo build --release --features onnx
```

  - Batch extraction is more efficient for multiple articles
  - Site profiles improve extraction for frequently visited sites

# Troubleshooting
## Import Error

```python
# If you get import errors, rebuild:
cd crates/article-extractor-py
maturin develop --release
```

## Model Loading Error

```python
# Check model path
import os
print(os.environ.get('ARTICLE_EXTRACTOR_MODEL_PATH'))

# Verify file exists
from pathlib import Path
model_path = Path("model.onnx")
print(f"Exists: {model_path.exists()}")
```

## Training Issues

  - Ensure you have enough HTML samples (at least 100+)
  - Check HTML files are readable
  - Monitor disk space for checkpoints

# Examples
See examples/ directory for complete examples:

  - extract_example.py - Basic extraction
  - complete_workflow.py - Training and extraction
  - batch_processing.py - Batch extraction

# API Reference
See Python docstrings:

```python
from article_extractor import RustArticleExtractor
help(RustArticleExtractor)
```

