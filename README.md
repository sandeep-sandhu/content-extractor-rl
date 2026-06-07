# Content Extractor RL

[![Crates.io](https://img.shields.io/crates/v/content-extractor-rl.svg)](https://crates.io/crates/content-extractor-rl)
[![docs.rs](https://img.shields.io/docsrs/content-extractor-rl)](https://docs.rs/content-extractor-rl)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![CI](https://github.com/sandeep-sandhu/content-extractor-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/sandeep-sandhu/content-extractor-rl/actions)

A high-performance Rust library for extracting article content from HTML pages. It combines a supervised content-node classifier (the "hybrid" selector) with Deep Reinforcement Learning (Dueling DQN with prioritized experience replay; experimental PPO/SAC) that tunes extraction parameters, scored by token F1 against ground-truth article text. A zero-dependency heuristic baseline fallback, site-specific profile memory, and curriculum learning round it out.

## Features

- **DQN-based extraction** — Dueling DQN with prioritized experience replay navigates the DOM tree to select the best content node, observing real per-candidate DOM features (word/link density, tag type, depth, Readability-style class/id signals)
- **Hybrid node classification** — a supervised content-node classifier selects the article node (labels derived automatically from ground-truth text), while RL tunes the continuous extraction parameters; falls back to a Readability-style heuristic when untrained
- **Ground-truth reward** — training scores extractions by token F1 against the labelled article text, not a self-referential proxy
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
- [Pre-trained Models](#pre-trained-models)
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

### Hybrid extraction (no trained model required)

`extract_article` is the one-call entry point. With `agent = None` it uses the
supervised/heuristic **hybrid** node selector (Readability-style features) plus
block-level filtering — strictly better than the raw baseline, and needs no
model:

```rust
use content_extractor_rl::{Config, extract_article, Result};

fn main() -> Result<()> {
    let config = Config::default();
    let html = std::fs::read_to_string("article.html")?;

    let article = extract_article(&html, "https://example.com/article", &config, None)?;

    println!("Method:  {}", article.method);          // "hybrid" (or "baseline" fallback)
    println!("Title:   {}", article.title.unwrap_or_default());
    println!("Quality: {:.3}", article.quality_score);
    println!("Content: {}", article.content);
    Ok(())
}
```

### RL-based extraction with a trained model

Pass a loaded agent to `extract_article`; it runs the agent greedily through the
extraction environment and returns the best result. `AgentFactory::load`
auto-detects the algorithm (DQN/PPO/SAC) from the model file:

```rust
use content_extractor_rl::{Config, AgentFactory, extract_article, get_device, Result};
use std::path::Path;

fn extract_with_model(html: &str, url: &str, model_path: &str) -> Result<String> {
    let config = Config::default();
    let device = get_device(); // CPU, or CUDA when built with --features cuda

    let agent = AgentFactory::load(
        Path::new(model_path),
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        &device,
    )?;

    let article = extract_article(html, url, &config, Some(agent.as_ref()))?;
    Ok(article.content)
}
```

### Train a model on your own data

Training rewards extractions by **token F1 against the ground-truth article
text**. Provide it via `TrainingSample::with_ground_truth`; if you only have
`(html, url)` pairs, `TrainingSample::from((html, url))` trains against a
self-supervised quality proxy instead.

```rust
use content_extractor_rl::{Config, TrainingSample, train_with_improvements, Result};
use std::path::Path;

fn main() -> Result<()> {
    let config = Config::default();

    let samples: Vec<TrainingSample> = vec![
        TrainingSample::with_ground_truth(
            std::fs::read_to_string("page1.html")?,
            "https://example.com/1".to_string(),
            "The known-good article body text…".to_string(),
        ),
        // …or without ground truth:
        TrainingSample::from((
            std::fs::read_to_string("page2.html")?,
            "https://example.com/2".to_string(),
        )),
    ];

    let (agent, metrics) = train_with_improvements(&config, samples)?;

    println!("Episodes:     {}", metrics.episode_rewards.len());
    println!("Best quality: {:.3}", metrics.best_avg_quality);

    agent.save(Path::new("models/my_model.safetensors"))?;
    Ok(())
}
```

### Train & use the hybrid node classifier

The supervised classifier learns which DOM node is the article body (labels are
derived automatically from ground-truth text). Train it, save it, then load it
for extraction:

```rust
use content_extractor_rl::{
    Config, TrainingSample, train_classifier, NodeClassifier, HybridExtractor,
    extract_article_hybrid, get_device, Result,
};
use std::path::Path;

fn main() -> Result<()> {
    let config = Config::default();
    let device = get_device();

    // Samples must carry ground-truth text for the classifier to learn from.
    let samples: Vec<TrainingSample> = load_labelled_samples();

    let (classifier, loss) = train_classifier(&samples, &config, 300, 1e-2, &device)?;
    println!("Final BCE loss: {loss:.4}");
    classifier.save(Path::new("models/node_classifier.safetensors"))?;

    // …later, load and extract:
    let clf = NodeClassifier::load(Path::new("models/node_classifier.safetensors"), &device, 1e-3)?;
    let hybrid = HybridExtractor::with_classifier(clf, config.stopwords.clone());
    let html = std::fs::read_to_string("article.html")?;
    let article = extract_article_hybrid(&html, "https://example.com/post", &config, &hybrid)?;
    println!("{}", article.content);
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
│   │   ├── text_utils.rs           ← Tokenisation, quality metrics, token F1
│   │   ├── node_features.rs        ← Real per-candidate DOM features + extraction params
│   │   ├── node_classifier.rs      ← Supervised content-node classifier + HybridExtractor
│   │   ├── environment.rs          ← RL environment (real state/action/reward MDP)
│   │   ├── replay_buffer.rs        ← Prioritised experience replay
│   │   ├── reward.rs               ← Legacy multi-component reward calculator
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
| State space | 300-dimensional float vector — real per-candidate DOM features (word/link density, stopword ratio, tag type, depth, Readability class/id signals) + global document + selection state |
| Action space | 16 discrete actions (select candidate 0-9, navigate parent/siblings, expand/contract, terminate) + 6 continuous parameters (block filtering) |
| Reward | Token F1 of the extracted text vs. the ground-truth article (falls back to a text-quality proxy when no ground truth is supplied) |
| Episode length | Up to `max_steps_per_episode` (default 20) |

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

`--model` is optional. Without it, extraction uses the hybrid heuristic node
selector (no model required); with it, the trained RL agent drives selection.

```bash
# Hybrid heuristic (no model)
content-extractor-rl extract \
    --html-file article.html \
    --url https://example.com/article \
    --output result.json

# With a trained RL model (DQN/PPO/SAC auto-detected)
content-extractor-rl extract \
    --html-file article.html \
    --url https://example.com/article \
    --model models/DuelingDQN.safetensors \
    --output result.json

# With a trained node classifier (hybrid selector)
content-extractor-rl extract \
    --html-file article.html \
    --url https://example.com/article \
    --classifier models/node_classifier.safetensors \
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

#### Train the node classifier (hybrid selector)

Trains the supervised content-node classifier from HTML + paired ground-truth
JSON, and writes a `.safetensors` file usable via `extract --classifier`:

```bash
content-extractor-rl train-classifier \
    --data-dir ./training_html \
    --output models/node_classifier.safetensors \
    --epochs 300 \
    --learning-rate 0.01
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
use content_extractor_rl::{Config, TrainingSample, train_with_improvements, Result};
use std::path::Path;

fn main() -> Result<()> {
    let mut config = Config::default();
    // Increase batch size for better stability
    config.batch_size = 1024;
    config.learning_rate = 3e-4;   // f64; ~3e-4 is a safe default
    config.gamma = 0.95;

    // load_html_dir returns Vec<TrainingSample> (read the JSON `text` field for
    // ground truth, see "Training data format" below).
    let samples: Vec<TrainingSample> = load_html_dir("./training_data")?;

    let (agent, metrics) = train_with_improvements(&config, samples)?;
    println!("Training complete. Best quality: {:.3}", metrics.best_avg_quality);

    // Save model
    let model_path = Path::new("models/my_model.safetensors");
    agent.save(model_path)?;

    // Save with full metadata
    agent.save_with_metadata(
        model_path,
        config.num_episodes,
        std::collections::HashMap::from([
            ("learning_rate".to_string(), config.learning_rate),
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

2. **Ground-truth reward** — when a sample carries ground-truth text, the reward is the token F1 of the extraction against it (`TextUtils::token_f1`), so the agent is optimised directly toward the labelled article. Supply it via `TrainingSample::with_ground_truth` (the CLI reads it from the JSON `text` field automatically). The continuous action params (`min_block_words`, `max_block_link_density`) control block-level filtering and are tuned by the policy.

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

## Pre-trained Models

Three models are included in the `models/` directory of this repository, each trained for 10,000 episodes on a corpus of 15,000 HTML pages from diverse news and article domains.

### Available ONNX Models

| File | Algorithm | Episodes | Best Quality | File Size | Trained On | Notes |
|------|-----------|----------|-------------|-----------|------------|-------|
| `DuelingDQN.onnx` | Dueling DQN | 10,000 | 0.8255 | 1.29 MB | CPU | Production-ready, stable training |
| `PPO.onnx` | PPO (Actor-Critic) | 10,000 | 0.8445 | 1.26 MB | GPU (CUDA) | Experimental; 36 h training run |
| `SAC.onnx` | SAC (Twin-Q) | 10,000 | 0.8445 | 3.51 MB | CPU | Experimental; see algorithm notes |

All three files are also available in SafeTensors format alongside best-hyperparameter JSON files for each algorithm.

### Hyperparameters used for training

| Hyperparameter | DuelingDQN | PPO | SAC |
|---|---|---|---|
| `learning_rate` | 0.002526 | 0.008220 | 0.005867 |
| `batch_size` | 2048 | 512 | 8192 |
| `gamma` | 0.856 | 0.858 | 0.988 |
| `epsilon_decay` | 0.9851 | 0.9859 | 0.9959 |
| `hidden_layers` | [512, 512, 256, 128] | [512, 512, 256, 128] | [1024, 512, 256] |
| `layer_norm` | no | yes | yes |

Hyperparameters were found via TPE Bayesian optimisation (`content-extractor-rl tune`). The full search results are in `output/`.

### Using a pre-trained model

```rust
use content_extractor_rl::{AgentFactory, Config, extract_article, get_device, Result};
use std::path::Path;

fn main() -> Result<()> {
    let config = Config::default();
    let device = get_device(); // CPU, or CUDA when built with --features cuda

    // Algorithm (DQN/PPO/SAC) is auto-detected from the model metadata.
    let agent = AgentFactory::load(
        Path::new("models/DuelingDQN.safetensors"),
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        &device,
    )?;

    let html = std::fs::read_to_string("page.html")?;
    let article = extract_article(&html, "https://example.com/post", &config, Some(agent.as_ref()))?;
    println!("{}", article.content);
    Ok(())
}
```

Use `DuelingDQN.onnx` for all production workloads. The PPO and SAC models are experimental and provided for research comparison only.

### Algorithm notes

- **DuelingDQN** — fully stable training run; no warnings. Best choice for production inference.
- **PPO** — stable training run on CUDA GPU (36.2 hours). Quality plateaus around episode 7,500.
- **SAC** — the automatic entropy temperature (`log_alpha`) was not receiving gradient updates in earlier code due to a disconnected computation graph (constant tensor vs. the `Var` leaf). This has been fixed in v0.1.3. The included `SAC.onnx` was trained prior to the fix and should be considered a baseline rather than a tuned model.

### Downloading via the CLI

```bash
# Download the latest general-purpose DQN model
content-extractor-rl download-model --output models/

# List available models
content-extractor-rl download-model --list
```

---

## API Reference

### Core types

```rust
// Main configuration (selected fields)
pub struct Config {
    pub state_dim: usize,              // 300 — state vector dimension
    pub num_discrete_actions: usize,   // 16 — discrete action count
    pub num_continuous_params: usize,  // 6  — continuous parameter count
    pub num_candidate_nodes: usize,    // 10 — candidate DOM nodes scored
    pub learning_rate: f64,            // default: 3e-4
    pub batch_size: usize,             // default: 512
    pub gamma: f64,                    // default: 0.95
    pub epsilon_start: f64,            // default: 1.0
    pub epsilon_end: f64,              // default: 0.05
    pub epsilon_decay: f64,            // default: 0.995
    pub replay_buffer_size: usize,     // default: 100_000
    pub target_update_freq: usize,     // default: 500
    pub max_steps_per_episode: usize,  // default: 20
    pub num_episodes: usize,           // default: 10_000
    // ...
}

// Extraction result
pub struct ExtractedArticle {
    pub url: String,
    pub title: Option<String>,
    pub date: Option<String>,
    pub content: String,
    pub quality_score: f32,
    pub method: String,   // "rl" | "hybrid" | "baseline" (+ "+profile" in batch)
    pub xpath: Option<String>,
}

// One-call extraction (RL agent when Some, else hybrid/heuristic, baseline fallback)
pub fn extract_article(
    html: &str, url: &str, config: &Config, agent: Option<&dyn RLAgent>,
) -> Result<ExtractedArticle>;
```

### Training

```rust
// A training example; ground truth enables token-F1 reward.
pub struct TrainingSample { pub html: String, pub url: String, pub ground_truth_text: Option<String> }
impl TrainingSample {
    pub fn with_ground_truth(html: String, url: String, ground_truth_text: String) -> Self;
}
impl From<(String, String)> for TrainingSample { /* ground truth = None */ }

// Standard training loop
pub fn train_standard(
    config: &Config,
    samples: Vec<TrainingSample>,
) -> Result<(Box<dyn RLAgent>, TrainingMetrics)>;

// Training with curriculum learning + ground-truth reward
pub fn train_with_improvements(
    config: &Config,
    samples: Vec<TrainingSample>,
) -> Result<(Box<dyn RLAgent>, TrainingMetrics)>;

pub struct TrainingMetrics {
    pub episode_rewards: Vec<f32>,
    pub episode_qualities: Vec<f32>,   // per-episode token F1 (or quality proxy)
    pub episode_losses: Vec<f32>,
    pub best_avg_quality: f32,
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
                  num_params: usize, gamma: f32, lr: f64, device: &Device) -> Result<Box<dyn RLAgent>>;
    // Algorithm is auto-detected from the saved model's metadata.
    pub fn load(path: &Path, state_dim: usize, num_actions: usize,
                num_params: usize, device: &Device) -> Result<Box<dyn RLAgent>>;
}
```

### Hybrid node classifier

```rust
use std::collections::HashSet;

// Supervised content-node classifier (MLP over NodeFeatures).
pub struct NodeClassifier { /* ... */ }
impl NodeClassifier {
    pub fn new(device: &Device, lr: f64) -> Result<Self>;
    pub fn train_batch(&mut self, features: &[NodeFeatures], labels: &[f32]) -> Result<f32>;
    pub fn score_batch(&self, features: &[NodeFeatures]) -> Result<Vec<f32>>;
    pub fn select_best(&self, features: &[NodeFeatures]) -> Result<Option<usize>>;
}

// Labels for free: 1.0 for the candidate whose text best matches the ground truth.
pub fn label_from_f1(contents: &[CandidateContent], gt: &str, stopwords: &HashSet<String>) -> Option<Vec<f32>>;

// End-to-end: classifier (or heuristic) picks the node, params drive extraction.
pub struct HybridExtractor { /* ... */ }
impl HybridExtractor {
    pub fn heuristic(stopwords: HashSet<String>) -> Self;               // no model
    pub fn with_classifier(clf: NodeClassifier, stopwords: HashSet<String>) -> Self;
    pub fn extract(&self, html: &str, num_candidates: usize, params: &ExtractionParams)
        -> Result<Option<HybridExtraction>>;
}
```

### Baseline extractor

```rust
use std::collections::HashSet;

pub struct BaselineExtractor {
    // stopword-density based heuristic, no neural network required
}

impl BaselineExtractor {
    pub fn new(stopwords: HashSet<String>) -> Self;
    pub fn extract(&self, html: &str) -> Result<ExtractionResult>;
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
| `DuelingDQN` | **Production-ready** | Fully tested, checkpoint resume, prioritised replay, gradient clipping |
| `PPO` | Experimental | Actor-critic structure working; GAE not fully verified |
| `SAC` | Experimental | Twin-Q + automatic entropy tuning; now has gradient clipping (fixes the earlier NaN divergence under high learning rates) |
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

The `content-extractor-rl-py` crate compiles to a native Python module named
`content_extractor_rl_rs`, built with [Maturin](https://maturin.rs/).

### Building / installing the wheel

```bash
cd crates/content-extractor-rl-py

# Option A — develop install into the current venv (fastest for iterating)
maturin develop --release

# Option B — build a distributable wheel, then pip install it
maturin build --release
pip install ../../target/wheels/content_extractor_rl_rs-*.whl

# With CUDA support:
maturin build --release --features cuda
```

> Requires a Rust toolchain and Python 3.8+. On aarch64 (e.g. Raspberry Pi 5),
> the candle dependency needs FP16 SIMD — the repo's `.cargo/config.toml` sets
> `target-cpu=native` to enable it.

### Usage

```python
from content_extractor_rl_rs import RustArticleExtractor

# No model -> hybrid heuristic selection (no training required).
extractor = RustArticleExtractor()

# ...or load a trained RL model (DQN/PPO/SAC auto-detected):
extractor = RustArticleExtractor(model="models/DuelingDQN.safetensors")

html = open("article.html").read()
result = extractor.extract(html, "https://example.com/article")
print(result["method"])         # "rl" | "hybrid" | "baseline"
print(result["title"])
print(result["quality_score"])
print(result["content"])

# Batch extraction: list of (html, url) tuples -> {"articles": [...]}
batch = extractor.extract_batch([(html, "https://example.com/article")])
for art in batch["articles"]:
    print(art["url"], art["quality_score"])

# Check hardware
print("CUDA:", extractor.check_cuda_available())
```

### Training from Python

```python
from content_extractor_rl_rs import RustArticleExtractor

extractor = RustArticleExtractor()

# html_samples is a list of (html, url) pairs.
samples = [(open(p).read(), url) for p, url in my_dataset]

metrics = extractor.train(samples, episodes=5000, improved=True)
print("Best quality:", metrics["best_avg_quality"])
```

> Note: the Python `train` accepts `(html, url)` pairs and trains against the
> self-supervised quality proxy. For token-F1-against-ground-truth training, use
> the Rust API (`TrainingSample::with_ground_truth`) or the CLI `train` command,
> which reads the ground-truth `text` from the paired JSON files.

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
