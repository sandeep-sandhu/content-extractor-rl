# Content Extractor RL

[![Crates.io](https://img.shields.io/crates/v/content-extractor-rl.svg)](https://crates.io/crates/content-extractor-rl)
[![docs.rs](https://img.shields.io/docsrs/content-extractor-rl)](https://docs.rs/content-extractor-rl)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![CI](https://github.com/sandeep-sandhu/content-extractor-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/sandeep-sandhu/content-extractor-rl/actions)

A high-performance Rust library for extracting article content from HTML pages. Uses Deep Reinforcement Learning (Dueling DQN with prioritized experience replay) with a heuristic baseline fallback, site-specific profile memory, and curriculum learning.

## Features

- **DQN-based extraction** — Dueling DQN with prioritized experience replay navigates the DOM tree to select the best content node
- **Baseline fallback** — stopword-density heuristic runs with zero dependencies on a trained model
- **Site profile memory** — per-domain XPath patterns learned and reused across sessions
- **Curriculum learning** — training progresses from simple to complex HTML layouts automatically
- **Hyperparameter optimization** — grid search and Tree-structured Parzen Estimator (TPE) Bayesian optimization
- **Multiple RL algorithms** — DuelingDQN (production-ready), PPO and SAC (experimental)
- **CUDA acceleration** — optional GPU support via the `cuda` feature flag
- **SafeTensors + ONNX serialization** — trained models saved in portable formats
- **MLflow integration** — optional experiment tracking via the `mlflow-rs` feature
- **Python bindings** — PyO3-based bindings for Python consumers (`content-extractor-rl-py`)
- **CLI tool** — full-featured `content-extractor-rl` binary for training and extraction

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [CLI Tool](#cli-tool)
- [Training Custom Models](#training-custom-models)
- [Downloading Pre-trained Weights](#downloading-pre-trained-weights)
- [API Reference](#api-reference)
- [Feature Flags](#feature-flags)
- [Performance Notes](#performance-notes)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
content-extractor-rl = "0.1"
```

With CUDA support:

```toml
[dependencies]
content-extractor-rl = { version = "0.1", features = ["cuda"] }
```

With MLflow experiment tracking:

```toml
[dependencies]
content-extractor-rl = { version = "0.1", features = ["mlflow-rs"] }
```

### System Requirements

| Requirement | Version |
|-------------|---------|
| Rust | 1.74+ |
| CUDA (optional) | 11.8+ (for `cuda` feature) |
| Python (optional) | 3.8+ (for Python bindings) |

On Ubuntu/Debian, install HTML parsing dependencies:

```bash
sudo apt-get install libssl-dev pkg-config
```

---

## Quick Start

### Baseline extraction (no trained model required)

```rust
use content_extractor_rl::{Config, BaselineExtractor, Result};

fn main() -> Result<()> {
    let config = Config::default();
    let extractor = BaselineExtractor::new(config.stopwords.clone());

    let html = std::fs::read_to_string("article.html")?;
    let article = extractor.extract(&html)?;

    println!("Title:   {}", article.title.unwrap_or_default());
    println!("Quality: {:.3}", article.quality_score);
    println!("Content: {}…", &article.content[..200]);
    Ok(())
}
```

### RL-based extraction with a trained model

```rust
use content_extractor_rl::{
    Config, AgentFactory, AlgorithmType, ArticleExtractionEnvironment,
    BaselineExtractor, Result,
};
use std::path::Path;

fn extract_with_model(html: &str, url: &str, model_path: &str) -> Result<String> {
    let config = Config::default();
    let device = content_extractor_rl::get_device(false)?;

    // Load a trained DQN agent
    let agent = AgentFactory::load(
        Path::new(model_path),
        AlgorithmType::DuelingDQN,
        &device,
    )?;

    // Run the RL extraction environment
    let mut env = ArticleExtractionEnvironment::new(config.clone());
    let mut state = env.reset(html, url);
    let mut done = false;

    while !done {
        let (action, params) = agent.select_action(&state, 0.0)?; // epsilon=0 = greedy
        let (next_state, _reward, is_done, _info) = env.step((action, params))?;
        state = next_state;
        done = is_done;
    }

    Ok(env.get_best_extraction().content)
}
```

### Train a model on your own data

```rust
use content_extractor_rl::{Config, train_with_improvements, Result};

fn main() -> Result<()> {
    let config = Config::default();

    // Each sample is (html_string, url_string)
    let samples: Vec<(String, String)> = load_your_html_samples();

    let (agent, metrics) = train_with_improvements(&config, samples)?;

    println!("Episodes: {}", metrics.episodes);
    println!("Best quality: {:.3}", metrics.best_avg_quality);

    agent.save(std::path::Path::new("model.safetensors"))?;
    Ok(())
}
```

---

## Architecture

```
content-extractor-rl (workspace)
├── crates/content-extractor-rl        ← Rust library (this crate)
│   ├── src/
│   │   ├── lib.rs                  ← Public API & re-exports
│   │   ├── config.rs               ← Configuration & env vars
│   │   ├── baseline_extractor.rs   ← Heuristic extraction
│   │   ├── html_parser.rs          ← DOM traversal, candidate extraction
│   │   ├── text_utils.rs           ← Tokenisation, quality metrics
│   │   ├── environment.rs          ← RL environment (state/action/reward)
│   │   ├── replay_buffer.rs        ← Prioritised experience replay
│   │   ├── reward.rs               ← Multi-component reward calculator
│   │   ├── curriculum.rs           ← Curriculum learning manager
│   │   ├── models.rs               ← Dueling DQN network (Candle)
│   │   ├── agents/
│   │   │   ├── mod.rs              ← RLAgent trait & AgentFactory
│   │   │   ├── dqn_agent.rs        ← Dueling DQN (production-ready)
│   │   │   ├── ppo_agent.rs        ← PPO actor-critic (experimental)
│   │   │   └── sac_agent.rs        ← SAC twin-Q (experimental)
│   │   ├── training.rs             ← Training loops
│   │   ├── hyperparameter.rs       ← Grid search
│   │   ├── hyperparameter_tuner.rs ← TPE Bayesian optimisation
│   │   ├── site_profile.rs         ← Per-domain pattern memory
│   │   ├── checkpoint.rs           ← Save/resume checkpoints
│   │   ├── evaluation/             ← Ground-truth & algorithm comparison
│   │   └── plotting.rs             ← Training visualisation
│   └── tests/                      ← Integration tests
├── crates/content-extractor-rl-cli    ← CLI binary
└── crates/content-extractor-rl-py     ← Python bindings (PyO3/Maturin)
```

### RL Environment

| | Detail |
|---|---|
| State space | 300-dimensional float vector (document features + candidate node features + domain history) |
| Action space | 16 discrete actions (select candidate 0-9, navigate parent/siblings, terminate) + 6 continuous parameters |
| Reward | Multi-component: text quality (50%), length bonus, structure bonus, improvement over baseline |
| Episode length | Up to 50 steps |

### Neural Network

The Dueling DQN network architecture:

```
Input (300) → FC(512) → LN → ReLU → FC(256) → LN → ReLU → FC(128) → LN → ReLU
                                                                           │
                              ┌────────────────────────────────────────────┤
                              │                                            │
                       Value stream                               Advantage stream
                       FC(64) → FC(1)                            FC(64) → FC(16)
                              │                                            │
                              └──────── Q(s,a) = V(s) + A(s,a) - mean(A) ┘
                                                         │
                                               Continuous params
                                               FC(128) → FC(6) → tanh
```

---

## CLI Tool

The `content-extractor-rl-cli` crate installs as the `content-extractor-rl` binary.

```bash
cargo install content-extractor-rl-cli
```

### Commands

#### Extract a single article

```bash
content-extractor-rl extract \
    --html-file article.html \
    --url https://example.com/article \
    --model models/dqn_model.safetensors \
    --output result.json
```

#### Batch extract from a directory

```bash
content-extractor-rl extract-batch \
    --archive-dir ./html_archive \
    --model models/dqn_model.safetensors \
    --output-dir ./extracted \
    --max-files 1000
```

#### Train a model

```bash
# Standard training (DQN, 5000 episodes)
content-extractor-rl train \
    --data-dir ./training_html \
    --episodes 5000 \
    --algorithm dqn

# Improved training with curriculum learning
content-extractor-rl train \
    --data-dir ./training_html \
    --episodes 10000 \
    --improved \
    --algorithm dqn \
    --models-dir ./models

# Auto-hyperparameter search before training
content-extractor-rl train \
    --data-dir ./training_html \
    --episodes 10000 \
    --improved \
    --auto-hyperparams
```

#### Hyperparameter tuning (TPE Bayesian optimisation)

```bash
content-extractor-rl tune \
    --data-dir ./training_html \
    --trials 50 \
    --episodes-per-trial 500 \
    --algorithm dqn \
    --output-dir ./tuning_results

# Resume an interrupted tuning run
content-extractor-rl tune \
    --data-dir ./training_html \
    --trials 50 \
    --resume \
    --output-dir ./tuning_results
```

#### Evaluate extraction quality against ground truth

```bash
content-extractor-rl evaluate \
    --data-dir ./ground_truth_json \
    --model models/dqn_model.safetensors
```

#### Compare multiple algorithms

```bash
content-extractor-rl compare \
    --data-dir ./test_html \
    --algorithms dqn,ppo,sac
```

---

## Training Custom Models

### Preparing training data

Collect raw HTML pages from the websites you care about. Place them in a flat directory — the filename should contain the domain for site-profile tracking:

```
training_data/
├── reuters_com_article_001.html
├── reuters_com_article_002.html
├── bbc_co_uk_article_001.html
├── techcrunch_com_post_001.html
└── ...
```

**Recommended minimum:** 100 HTML files per domain, 500+ total.

### Training from Rust code

```rust
use content_extractor_rl::{Config, train_with_improvements, Result};
use std::path::Path;

fn main() -> Result<()> {
    let mut config = Config::default();
    // Increase batch size for better stability
    config.batch_size = 1024;
    config.learning_rate = 3e-4;
    config.gamma = 0.95;

    let samples = load_html_dir("./training_data")?;

    let (agent, metrics) = train_with_improvements(&config, samples)?;
    println!("Training complete. Best quality: {:.3}", metrics.best_avg_quality);

    // Save model
    let model_path = Path::new("models/my_model.safetensors");
    agent.save(model_path)?;

    // Save with full metadata
    agent.save_with_metadata(
        model_path,
        metrics.episodes,
        std::collections::HashMap::from([
            ("learning_rate".to_string(), config.learning_rate as f64),
            ("batch_size".to_string(), config.batch_size as f64),
        ])
    )?;
    Ok(())
}
```

### Training for specific news/article websites

To customise the model for specific websites, the key levers are:

1. **Site profiles** — the library automatically builds per-domain XPath profiles as it trains. After training, save the site profile directory:

```bash
export ARTICLE_EXTRACTOR_SITE_PROFILES=./site_profiles
content-extractor-rl train --data-dir ./training_data --episodes 5000 --improved
# site_profiles/ now contains per-domain learned patterns
```

2. **Reward shaping** — the `ImprovedRewardCalculator` scores extractions on text quality. If a site uses non-standard markup, you can adjust quality thresholds in the `Config`:

```rust
config.min_word_threshold = 50;   // minimum words to count as an article
config.stopword_weight = 2.5;     // reward stopword-rich paragraphs more
```

3. **Curriculum difficulty** — for sites with complex layouts (heavy JavaScript-rendered content, infinite scroll), start with simpler pages. The `CurriculumManager` handles this automatically when you use `train_with_improvements`.

4. **Pre-training workflow** for a set of target sites:

```bash
# Step 1: Tune hyperparameters on a representative sample
content-extractor-rl tune \
    --data-dir ./training_sample \
    --trials 30 \
    --episodes-per-trial 300 \
    --output-dir ./tuning

# Step 2: Train with the best hyperparameters
content-extractor-rl train \
    --data-dir ./full_training_data \
    --episodes 15000 \
    --improved \
    --hyperparams ./tuning/best_hyperparams_dqn.json \
    --models-dir ./models

# Step 3: Verify quality
content-extractor-rl evaluate \
    --data-dir ./validation_data \
    --model ./models/best_model.safetensors
```

### Training data format for ground-truth evaluation

To use `evaluate` and measure accuracy against known-good extractions, provide JSON files alongside HTML:

```json
{
  "url": "https://example.com/article",
  "title": "Article headline here",
  "text": "Full article body text goes here...",
  "author": "Author Name",
  "pubDate": "2024-01-15"
}
```

---

## Downloading Pre-trained Weights

Pre-trained model weights are provided as GitHub Release attachments. These are general-purpose models trained on a diverse corpus of news and blog articles.

### Downloading via the CLI

```bash
# Download the latest general-purpose DQN model
content-extractor-rl download-model --output models/

# List available models
content-extractor-rl download-model --list
```

### Manual download

Go to [GitHub Releases](https://github.com/sandeepsandhu/content-extractor-rl/releases) and download:

| File | Description | Size |
|------|-------------|------|
| `dqn_general_v1.safetensors` | General news/blog articles | ~15 MB |
| `dqn_news_v1.safetensors` | Tuned for news sites (Reuters, BBC, etc.) | ~15 MB |
| `site_profiles_v1.tar.gz` | Site-specific XPath profiles | ~1 MB |

### Using a downloaded model

```rust
use content_extractor_rl::{AgentFactory, AlgorithmType, get_device, Result};
use std::path::Path;

fn main() -> Result<()> {
    let device = get_device(false)?; // false = CPU
    let agent = AgentFactory::load(
        Path::new("models/dqn_general_v1.safetensors"),
        AlgorithmType::DuelingDQN,
        &device,
    )?;
    // use agent for extraction...
    Ok(())
}
```

---

## API Reference

### Core types

```rust
// Main configuration
pub struct Config {
    pub state_dim: usize,            // 300 — state vector dimension
    pub num_actions: usize,          // 16 — discrete action count
    pub num_params: usize,           // 6  — continuous parameter count
    pub learning_rate: f32,          // default: 1e-4
    pub batch_size: usize,           // default: 512
    pub gamma: f32,                  // default: 0.95
    pub epsilon_start: f32,          // default: 1.0
    pub epsilon_end: f32,            // default: 0.01
    pub epsilon_decay: f32,          // default: 0.995
    pub replay_buffer_size: usize,   // default: 100_000
    pub target_update_freq: usize,   // default: 1_000
    pub max_steps_per_episode: usize,// default: 50
    // ...
}

// Extraction result
pub struct ExtractedArticle {
    pub url: String,
    pub title: Option<String>,
    pub date: Option<String>,
    pub content: String,
    pub quality_score: f32,
    pub method: String,   // "rl" | "baseline" | "site_profile"
    pub xpath: Option<String>,
}
```

### Training

```rust
// Standard training loop
pub fn train_standard(
    config: &Config,
    html_samples: Vec<(String, String)>,   // (html, url)
) -> Result<(Box<dyn RLAgent>, TrainingMetrics)>;

// Training with curriculum learning + improved rewards
pub fn train_with_improvements(
    config: &Config,
    html_samples: Vec<(String, String)>,
) -> Result<(Box<dyn RLAgent>, TrainingMetrics)>;

pub struct TrainingMetrics {
    pub episodes: usize,
    pub best_avg_quality: f32,
    pub avg_reward: f32,
    pub episode_rewards: Vec<f32>,
    pub episode_qualities: Vec<f32>,
    pub episode_losses: Vec<f32>,
}
```

### Agent interface

```rust
pub trait RLAgent: Send + Sync {
    fn select_action(&self, state: &[f32], epsilon: f32) -> Result<(usize, Vec<f32>)>;
    fn train_step(&mut self, replay_buffer: &mut PrioritizedReplayBuffer, batch_size: usize) -> Result<f32>;
    fn update_target_network(&mut self);
    fn save(&self, path: &Path) -> Result<()>;
    fn save_with_metadata(&self, path: &Path, episodes: usize, hyperparams: HashMap<String, f64>) -> Result<()>;
    fn algorithm_type(&self) -> AlgorithmType;
    fn get_info(&self) -> AgentInfo;
}

// Create an agent from scratch
pub struct AgentFactory;
impl AgentFactory {
    pub fn create(algo: AlgorithmType, state_dim: usize, num_actions: usize,
                  num_params: usize, gamma: f32, lr: f32, device: &Device) -> Result<Box<dyn RLAgent>>;
    pub fn load(path: &Path, algo: AlgorithmType, device: &Device) -> Result<Box<dyn RLAgent>>;
}
```

### Baseline extractor

```rust
pub struct BaselineExtractor {
    // stopword-density based heuristic, no neural network required
}

impl BaselineExtractor {
    pub fn new(stopwords: Vec<String>) -> Self;
    pub fn extract(&self, html: &str) -> Result<ExtractedArticle>;
}
```

### Hyperparameter optimisation

```rust
// TPE Bayesian optimisation
pub struct TPEOptimizer { ... }

impl TPEOptimizer {
    pub fn new(space: HyperparameterSpace) -> Self;
    pub fn optimize(
        &mut self,
        base_config: Config,
        samples: Vec<(String, String)>,
        n_trials: usize,
    ) -> Result<Hyperparameters>;
    pub fn save_state(&self, path: &Path) -> Result<()>;
    pub fn load_state(path: &Path) -> Result<Self>;
}

pub struct HyperparameterSpace {
    pub learning_rate: (f64, f64),       // (min, max)
    pub batch_size: Vec<usize>,
    pub gamma: (f64, f64),
    pub epsilon_decay: (f64, f64),
    pub priority_alpha: (f64, f64),
    pub priority_beta: (f64, f64),
    pub hidden_layer_sizes: Vec<Vec<usize>>,
    pub use_layer_norm: Vec<bool>,
    pub dropout: (f32, f32),
}
```

### Evaluation

```rust
pub struct GroundTruthEvaluator { ... }

impl GroundTruthEvaluator {
    pub fn evaluate(&self, extracted: &ExtractedArticle, ground_truth: &GroundTruthData) -> Result<EvaluationMetrics>;
}

pub struct EvaluationMetrics {
    pub text_f1: f32,
    pub text_precision: f32,
    pub text_recall: f32,
    pub title_match: f32,
    pub combined_quality: f32,
}
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ARTICLE_EXTRACTOR_MODEL_PATH` | — | Path to a saved model file |
| `ARTICLE_EXTRACTOR_SITE_PROFILES` | `./site_profiles` | Directory for per-domain profiles |
| `ARTICLE_EXTRACTOR_OUTPUT_DIR` | `./output` | Directory for extraction outputs |
| `ARTICLE_EXTRACTOR_DATA_DIR` | — | Training data directory |

---

## Feature Flags

| Flag | Default | Description |
|---|---|---|
| `cuda` | off | Enable CUDA GPU acceleration via `candle-core/cuda` |
| `mlflow-rs` | off | Enable MLflow experiment tracking |

---

## Algorithm Status

| Algorithm | Status | Notes |
|---|---|---|
| `DuelingDQN` | **Production-ready** | Fully tested, checkpoint resume, prioritised replay |
| `PPO` | Experimental | Actor-critic structure working; GAE not fully verified |
| `SAC` | Experimental | Twin-Q networks present; entropy tuning needs testing |
| `TD3` | Not implemented | Placeholder in `AlgorithmType` enum |
| `Rainbow` | Not implemented | Placeholder in `AlgorithmType` enum |

Use `AlgorithmType::DuelingDQN` for all production workloads.

---

## Performance Notes

- Baseline extraction runs in **< 5 ms** per page on any hardware.
- DQN inference (model loaded) runs in **10–30 ms** per page on CPU; **< 5 ms** on GPU.
- Training throughput on CPU: ~200–500 episodes/min depending on HTML complexity.
- Training throughput on A100 GPU: ~2000–5000 episodes/min with `--features cuda`.
- The replay buffer holds 100,000 experiences by default (adjust `Config::replay_buffer_size` for memory-constrained environments).
- `rayon`-based parallel extraction is available for batch workloads via `extract-batch`.

---

## Python Bindings

The `content-extractor-rl-py` crate provides a Python package built with [Maturin](https://maturin.rs/).

```bash
pip install content-extractor-rl-rs    # when published to PyPI
# or build from source:
cd crates/content-extractor-rl-py
maturin develop --release
```

```python
from content_extractor_rl_rs import RustArticleExtractor

extractor = RustArticleExtractor(model="models/dqn_general_v1.safetensors")
result = extractor.extract(html_content, "https://example.com/article")
print(result["content"])
print(result["quality_score"])
```

---

## Contributing

Contributions are welcome. Areas where help is most needed:

- Completing PPO and SAC agent training loops
- Expanding the test suite (especially for `environment.rs`)
- Ground-truth datasets for news domains
- ONNX export improvements

Please open an issue before submitting a large PR.

```bash
git clone https://github.com/sandeepsandhu/content-extractor-rl
cd content-extractor-rl
cargo test --all
```

---

## License

Licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
