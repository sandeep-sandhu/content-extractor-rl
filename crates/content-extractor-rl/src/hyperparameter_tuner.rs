//! Hyperparameter tuning using TPE (Tree-structured Parzen Estimator) with resume capability
// ============================================================================
// FILE: crates/content-extractor-rl/src/hyperparameter_tuner.rs
// ============================================================================

use crate::{AlgorithmType, Config, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use rand::Rng;
use tracing::{info, warn};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use crate::models::NetworkConfig;

/// hyperparameter search space with network architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    // Training hyperparameters
    pub learning_rate: (f64, f64),
    pub batch_size: Vec<usize>,
    pub gamma: (f64, f64),
    pub epsilon_decay: (f64, f64),
    pub priority_alpha: (f64, f64),
    pub priority_beta: (f64, f64),

    // Network architecture hyperparameters
    pub hidden_layer_sizes: Vec<Vec<usize>>,  // Different architectures to try
    pub value_hidden: Vec<usize>,
    pub advantage_hidden: Vec<usize>,
    pub use_layer_norm: Vec<bool>,
    pub dropout: (f32, f32),
}

impl Default for HyperparameterSpace {
    fn default() -> Self {
        Self {
            learning_rate: (1e-5, 1e-2),
            batch_size: vec![256, 512, 1024, 2048, 4096, 6144, 8192],
            gamma: (0.85, 0.99),
            epsilon_decay: (0.985, 0.999),
            priority_alpha: (0.35, 0.8),
            priority_beta: (0.3, 0.7),

            // Network architectures to try
            hidden_layer_sizes: vec![
                vec![256, 128],           // Small
                vec![512, 256, 128],      // Default
                vec![1024, 512, 256],     // Large
                vec![512, 512, 256, 128], // Deep
            ],
            value_hidden: vec![32, 64, 128, 192],
            advantage_hidden: vec![32, 64, 128, 192],
            use_layer_norm: vec![true, false],
            dropout: (0.0, 0.01),
        }
    }
}

/// Hyperparameter configuration
/// Enhanced hyperparameters including network config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    // Training hyperparameters
    pub learning_rate: f64,
    pub batch_size: usize,
    pub gamma: f64,
    pub epsilon_decay: f64,
    pub priority_alpha: f64,
    pub priority_beta: f64,

    // Network architecture
    pub network_config: NetworkConfig,

    pub timestamp: String,
    pub quality_score: f64,
}

impl Hyperparameters {
    /// Apply hyperparameters to config
    pub fn apply_to_config(&self, config: &mut Config) {
        config.learning_rate = self.learning_rate;
        config.batch_size = self.batch_size;
        config.gamma = self.gamma;
        config.epsilon_decay = self.epsilon_decay;
        config.priority_alpha = self.priority_alpha;
        config.priority_beta = self.priority_beta;

        // Apply network config
        config.state_dim = self.network_config.state_dim;
        config.num_discrete_actions = self.network_config.num_actions;
        config.num_continuous_params = self.network_config.num_params;
    }

    /// Save to algorithm-specific JSON file
    pub fn save_with_algorithm(&self, base_path: &Path, algorithm: AlgorithmType) -> Result<()> {
        let filename = format!("best_hyperparams_{}.json", algorithm.to_string().to_lowercase());
        let path = base_path.parent()
            .unwrap_or(base_path)
            .join(filename);

        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, json)?;

        info!("✓ Saved {} hyperparameters to: {}", algorithm, path.display());
        Ok(())
    }

    /// Load from algorithm-specific file
    pub fn load_for_algorithm(base_dir: &Path, algorithm: AlgorithmType) -> Result<Self> {
        let filename = format!("best_hyperparams_{}.json", algorithm.to_string().to_lowercase());
        let path = base_dir.join(&filename);

        if !path.exists() {
            return Err(crate::ExtractionError::ParseError(
                format!("Hyperparameters file not found: {}", path.display())
            ));
        }

        let json = std::fs::read_to_string(&path)?;
        let params:Hyperparameters = serde_json::from_str(&json)?;

        info!("✓ Loaded {} hyperparameters from: {}", algorithm, path.display());
        info!("  Settings:");
        info!("    learning_rate: {:.6}", params.learning_rate);
        info!("    batch_size: {}", params.batch_size);
        info!("    gamma: {:.3}", params.gamma);
        info!("    epsilon_decay: {:.6}", params.epsilon_decay);
        info!("    priority_alpha: {:.3}", params.priority_alpha);
        info!("    priority_beta: {:.3}", params.priority_beta);

        Ok(params)
    }

    /// Save to JSON file
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        info!("Saved hyperparameters to: {}", path.display());
        Ok(())
    }

    /// Load from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let params = serde_json::from_str(&json)?;
        info!("Loaded hyperparameters from: {}", path.display());
        Ok(params)
    }
}

/// Trial result from hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub trial_number: usize,
    pub hyperparameters: Hyperparameters,
    pub quality_score: f64,
    pub avg_reward: f64,
    pub duration_seconds: f64,
}

/// Optimizer state for resuming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub trials: Vec<TrialResult>,
    pub n_startup_trials: usize,
    pub space: HyperparameterSpace,
    pub best_trial: Option<usize>,
    pub timestamp: String,
}

impl OptimizerState {
    /// Save state to file
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        info!("Saved optimizer state to: {}", path.display());
        Ok(())
    }

    /// Load state from file
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let state = serde_json::from_str(&json)?;
        info!("Loaded optimizer state from: {}", path.display());
        Ok(state)
    }
}

/// TPE-based hyperparameter optimizer with resume capability
pub struct TPEOptimizer {
    space: HyperparameterSpace,
    trials: Vec<TrialResult>,
    n_startup_trials: usize,
    state_path: Option<PathBuf>,
}

impl TPEOptimizer {
    /// Create new TPE optimizer
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            space,
            trials: Vec::new(),
            n_startup_trials: 5, // As per requirement
            state_path: None,
        }
    }

    /// Run parallel hyperparameter optimization
    pub fn optimize_parallel(
        &mut self,
        n_trials: usize,
        episodes_per_trial: usize,
        html_samples: Vec<(String, String)>,
        base_config: &Config,
        n_workers: usize,
    ) -> Result<()> {
        info!("Starting parallel TPE optimization with {} workers", n_workers);

        // Configure rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| crate::ExtractionError::RuntimeError(e.to_string()))?;

        // Generate all trial parameters upfront (sequential, uses TPE logic)
        let mut all_trial_params = Vec::new();
        let mut rng = rand::rng();
        for trial_num in 0..n_trials {
            // Use random sampling for all trials in parallel mode
            let params = self.random_suggest(&mut rng);
            all_trial_params.push((trial_num, params));
        }

        // Shared state for collecting results
        let results = Arc::new(Mutex::new(Vec::new()));
        let completed_trials = Arc::new(Mutex::new(0usize));

        // Run trials in parallel
        pool.install(|| {
            all_trial_params.par_iter().for_each(|(trial_num, params)| {
                info!("Worker starting trial {}", trial_num);

                // Each worker gets its own config and data
                let mut trial_config = base_config.clone();
                params.apply_to_config(&mut trial_config);
                trial_config.num_episodes = episodes_per_trial;

                // Use CPU for parallel trials to avoid GPU contention
                trial_config.use_cpu_for_tuning = true;

                let trial_start = std::time::Instant::now();

                // Run training
                let result = crate::training::train_standard(&trial_config, html_samples.clone());

                match result {
                    Ok((_agent, metrics)) => {
                        let duration = trial_start.elapsed();

                        // Calculate quality
                        let window = metrics.episode_qualities.len().min(50);
                        let quality = if metrics.episode_qualities.len() >= window {
                            metrics.episode_qualities[metrics.episode_qualities.len() - window..]
                                .iter()
                                .sum::<f32>() / window as f32
                        } else if !metrics.episode_qualities.is_empty() {
                            metrics.episode_qualities.iter().sum::<f32>() /
                                metrics.episode_qualities.len() as f32
                        } else {
                            0.0
                        };

                        let avg_reward = if !metrics.episode_rewards.is_empty() {
                            let window = metrics.episode_rewards.len().min(50);
                            if metrics.episode_rewards.len() >= window {
                                metrics.episode_rewards[metrics.episode_rewards.len() - window..]
                                    .iter()
                                    .sum::<f32>() / window as f32
                            } else {
                                metrics.episode_rewards.iter().sum::<f32>() /
                                    metrics.episode_rewards.len() as f32
                            }
                        } else {
                            0.0
                        };

                        // Record result
                        let trial_result = TrialResult {
                            trial_number: *trial_num,
                            hyperparameters: Hyperparameters {
                                quality_score: quality as f64,
                                ..params.clone()
                            },
                            quality_score: quality as f64,
                            avg_reward: avg_reward as f64,
                            duration_seconds: duration.as_secs_f64(),
                        };

                        // Store result
                        {
                            let mut res = results.lock().unwrap();
                            res.push(trial_result);
                        }

                        {
                            let mut completed = completed_trials.lock().unwrap();
                            *completed += 1;
                            info!("Trial {} completed ({}/{}): quality={:.4}",
                          trial_num, *completed, n_trials, quality);
                        }
                    }
                    Err(e) => {
                        warn!("Trial {} failed: {}", trial_num, e);
                    }
                }
            });
        });

        // After all trials complete, update self with results
        let trial_results = results.lock().unwrap();
        for trial_result in trial_results.iter() {
            self.tell(trial_result.clone());
        }

        info!("Parallel optimization complete");
        Ok(())
    }

    /// Create optimizer with resume capability
    pub fn with_resume(space: HyperparameterSpace, state_path: PathBuf) -> Result<Self> {
        let mut optimizer = Self {
            space: space.clone(),
            trials: Vec::new(),
            n_startup_trials: 5,
            state_path: Some(state_path.clone()),
        };

        // Try to load existing state
        if state_path.exists() {
            info!("Resuming from saved state: {}", state_path.display());
            let state = OptimizerState::load(&state_path)?;
            optimizer.trials = state.trials;
            optimizer.space = state.space;
            optimizer.n_startup_trials = state.n_startup_trials;
            info!("Resumed with {} existing trials", optimizer.trials.len());
        }

        Ok(optimizer)
    }

    /// Random hyperparameter suggestion for initial trials
    pub fn random_suggest(&self, rng: &mut impl Rng) -> Hyperparameters {
        // Sample random network architecture
        let hidden_layers = self.space.hidden_layer_sizes
            .get(rng.random_range(0..self.space.hidden_layer_sizes.len()))
            .unwrap()
            .clone();

        let value_hidden = *self.space.value_hidden
            .get(rng.random_range(0..self.space.value_hidden.len()))
            .unwrap();

        let advantage_hidden = *self.space.advantage_hidden
            .get(rng.random_range(0..self.space.advantage_hidden.len()))
            .unwrap();

        let use_layer_norm = *self.space.use_layer_norm
            .get(rng.random_range(0..self.space.use_layer_norm.len()))
            .unwrap();

        let dropout = rng.random_range(self.space.dropout.0..self.space.dropout.1);

        Hyperparameters {
            learning_rate: rng.random_range(self.space.learning_rate.0..self.space.learning_rate.1),
            batch_size: *self.space.batch_size
                .get(rng.random_range(0..self.space.batch_size.len()))
                .unwrap(),
            gamma: rng.random_range(self.space.gamma.0..self.space.gamma.1),
            epsilon_decay: rng.random_range(self.space.epsilon_decay.0..self.space.epsilon_decay.1),
            priority_alpha: rng.random_range(self.space.priority_alpha.0..self.space.priority_alpha.1),
            priority_beta: rng.random_range(self.space.priority_beta.0..self.space.priority_beta.1),
            network_config: NetworkConfig {
                state_dim: 300,
                num_actions: 16,
                num_params: 6,
                hidden_layers,
                use_layer_norm,
                dropout,
                value_hidden,
                advantage_hidden,
            },
            timestamp: chrono::Utc::now().to_rfc3339(),
            quality_score: 0.0,
        }
    }

    /// Sample categorical choice (e.g., network architecture)
    #[allow(dead_code)]
    fn sample_tpe_categorical<T: Clone>(
        &self,
        good_values: Vec<&T>,
        _bad_values: Vec<&T>,
        choices: &[T],
        rng: &mut impl Rng,
    ) -> T {
        if good_values.is_empty() {
            return choices[rng.random_range(0..choices.len())].clone();
        }

        // Count frequency in good trials
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for _good_val in &good_values {
            for (i, _choice) in choices.iter().enumerate() {
                // This is a simplified comparison - in real code you'd need proper equality
                counts.entry(i).or_insert(0);
            }
        }

        // Weighted sampling based on good trial frequencies
        if counts.is_empty() {
            choices[rng.random_range(0..choices.len())].clone()
        } else {
            let total: usize = counts.values().sum();
            let r: f64 = rng.random::<f64>() * total as f64;
            let mut cumsum = 0.0;

            for (idx, count) in counts.iter() {
                cumsum += *count as f64;
                if r <= cumsum {
                    return choices[*idx].clone();
                }
            }
            choices[0].clone()
        }
    }

    /// Sample boolean parameter
    #[allow(dead_code)]
    fn sample_tpe_boolean(
        &self,
        good_values: Vec<bool>,
        _bad_values: Vec<bool>,
        rng: &mut impl Rng,
    ) -> bool {
        if good_values.is_empty() {
            return rng.random();
        }

        let true_count = good_values.iter().filter(|&&x| x).count();
        let probability = true_count as f64 / good_values.len() as f64;

        rng.random::<f64>() < probability
    }

    #[allow(dead_code)]
    fn good_trials(&self) -> Vec<TrialResult> {
        let quantile = 0.25;
        let mut sorted = self.trials.clone();
        sorted.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap());
        let n_good = (sorted.len() as f64 * quantile).ceil() as usize;
        sorted[..n_good].to_vec()
    }

    #[allow(dead_code)]
    fn bad_trials(&self) -> Vec<TrialResult> {
        let quantile = 0.25;
        let mut sorted = self.trials.clone();
        sorted.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap());
        let n_good = (sorted.len() as f64 * quantile).ceil() as usize;
        sorted[n_good..].to_vec()
    }

    /// Sample continuous parameter using TPE
    #[allow(dead_code)]
    fn sample_tpe_continuous(
        &self,
        good_values: Vec<f64>,
        _bad_values: Vec<f64>,
        bounds: (f64, f64),
        rng: &mut impl Rng,
    ) -> f64 {
        if good_values.is_empty() {
            return rng.random_range(bounds.0..bounds.1);
        }

        // Calculate mean and std for good and bad distributions
        let good_mean = good_values.iter().sum::<f64>() / good_values.len() as f64;
        let good_std = if good_values.len() > 1 {
            let variance = good_values.iter()
                .map(|x| (x - good_mean).powi(2))
                .sum::<f64>() / (good_values.len() - 1) as f64;
            variance.sqrt()
        } else {
            (bounds.1 - bounds.0) * 0.1
        };

        // Sample from good distribution (truncated normal)
        let value = self.sample_truncated_normal(good_mean, good_std, bounds, rng);
        value.clamp(bounds.0, bounds.1)
    }

    /// Sample discrete parameter using TPE
    #[allow(dead_code)]
    fn sample_tpe_discrete(
        &self,
        good_values: Vec<usize>,
        _bad_values: Vec<usize>,
        choices: &[usize],
        rng: &mut impl Rng,
    ) -> usize {
        if good_values.is_empty() {
            return *choices.get(rng.random_range(0..choices.len())).unwrap();
        }

        // Count frequency in good trials
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &val in &good_values {
            *counts.entry(val).or_insert(0) += 1;
        }

        // Choose based on frequency (weighted sampling)
        let total: usize = counts.values().sum();
        if total == 0 {
            return *choices.get(rng.random_range(0..choices.len())).unwrap();
        }

        let r: f64 = rng.random::<f64>() * total as f64;
        let mut cumsum = 0.0;

        for (&val, &count) in counts.iter() {
            cumsum += count as f64;
            if r <= cumsum {
                return val;
            }
        }

        // Fallback
        *good_values.last().unwrap()
    }

    /// Sample from truncated normal distribution
    #[allow(dead_code)]
    fn sample_truncated_normal(
        &self,
        mean: f64,
        std: f64,
        bounds: (f64, f64),
        rng: &mut impl Rng,
    ) -> f64 {
        use rand_distr::{Normal, Distribution};

        let normal = Normal::new(mean, std).unwrap_or_else(|_| Normal::new(mean, 0.1).unwrap());

        // Sample with rejection (max 100 attempts)
        for _ in 0..100 {
            let value = normal.sample(rng);
            if value >= bounds.0 && value <= bounds.1 {
                return value;
            }
        }

        // Fallback to clamped value
        mean.clamp(bounds.0, bounds.1)
    }

    /// Record trial result and save state
    pub fn tell(&mut self, trial: TrialResult) {
        info!(
            "Trial {}: quality={:.4}, lr={:.6}, batch={}, gamma={:.3}",
            trial.trial_number,
            trial.quality_score,
            trial.hyperparameters.learning_rate,
            trial.hyperparameters.batch_size,
            trial.hyperparameters.gamma
        );

        self.trials.push(trial);

        // Save state if path is configured
        if let Some(ref path) = self.state_path {
            let state = OptimizerState {
                trials: self.trials.clone(),
                n_startup_trials: self.n_startup_trials,
                space: self.space.clone(),
                best_trial: self.get_best_trial_idx(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            };

            if let Err(e) = state.save(path) {
                warn!("Failed to save optimizer state: {}", e);
            }
        }
    }

    /// Get best hyperparameters
    pub fn get_best(&self) -> Option<&Hyperparameters> {
        self.trials.iter()
            .max_by(|a, b| a.quality_score.partial_cmp(&b.quality_score).unwrap())
            .map(|t| &t.hyperparameters)
    }

    /// Get best trial index
    fn get_best_trial_idx(&self) -> Option<usize> {
        self.trials.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.quality_score.partial_cmp(&b.quality_score).unwrap())
            .map(|(idx, _)| idx)
    }

    /// Get number of trials completed
    pub fn num_trials(&self) -> usize {
        self.trials.len()
    }

    /// Save results with algorithm-specific filename
    pub fn save_results_for_algorithm(&self, output_dir: &Path, algorithm: AlgorithmType) -> Result<()> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("tuning_results_{}_{}.json",
                               algorithm.to_string().to_lowercase(),
                               timestamp
        );
        let path = output_dir.join(filename);

        let best_trial = self.get_best_trial_idx();

        let results = serde_json::json!({
            "algorithm": algorithm.to_string(),
            "n_trials": self.trials.len(),
            "best_quality": self.get_best().map(|h| h.quality_score).unwrap_or(0.0),
            "best_trial_number": best_trial.map(|i| self.trials[i].trial_number),
            "best_hyperparameters": self.get_best(),
            "all_trials": self.trials,
            "search_space": self.space,
        });

        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(&path, json)?;

        info!("✓ Saved {} tuning results to: {}", algorithm, path.display());
        Ok(())
    }

    /// Save optimization results
    pub fn save_results(&self, path: &Path) -> Result<()> {
        let best_trial = self.get_best_trial_idx();

        let results = serde_json::json!({
            "n_trials": self.trials.len(),
            "best_quality": self.get_best().map(|h| h.quality_score).unwrap_or(0.0),
            "best_trial_number": best_trial.map(|i| self.trials[i].trial_number),
            "best_hyperparameters": self.get_best(),
            "all_trials": self.trials,
            "search_space": self.space,
        });

        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(path, json)?;
        info!("Saved optimization results to: {}", path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_tpe_optimizer() {
        let space = HyperparameterSpace::default();
        let mut optimizer = TPEOptimizer::new(space);
        let mut rng = rand::rng();
        // Simulate some trials
        for i in 0..15 {
            let params = optimizer.random_suggest(&mut rng);
            let quality = 0.5 + i as f64 * 0.02; // Simulate improving quality

            let trial = TrialResult {
                trial_number: i,
                hyperparameters: Hyperparameters {
                    quality_score: quality,
                    ..params
                },
                quality_score: quality,
                avg_reward: quality * 2.0 - 1.0,
                duration_seconds: 100.0,
            };

            optimizer.tell(trial);
        }

        let best = optimizer.get_best().unwrap();
        assert!(best.quality_score > 0.7);
    }

    #[test]
    fn test_optimizer_resume() {
        let temp_dir = TempDir::new().unwrap();
        let state_path = temp_dir.path().join("optimizer_state.json");

        let space = HyperparameterSpace::default();

        // First session
        {
            let mut optimizer = TPEOptimizer::with_resume(space.clone(), state_path.clone()).unwrap();
            let mut rng = rand::rng();
            for i in 0..5 {
                let params = optimizer.random_suggest(&mut rng);
                let trial = TrialResult {
                    trial_number: i,
                    hyperparameters: Hyperparameters {
                        quality_score: 0.5 + i as f64 * 0.1,
                        ..params
                    },
                    quality_score: 0.5 + i as f64 * 0.1,
                    avg_reward: 0.0,
                    duration_seconds: 100.0,
                };
                optimizer.tell(trial);
            }

            assert_eq!(optimizer.num_trials(), 5);
        }

        // Resume session
        {
            let mut optimizer = TPEOptimizer::with_resume(space, state_path).unwrap();
            assert_eq!(optimizer.num_trials(), 5);
            let mut rng = rand::rng();
            // Continue with more trials
            for i in 5..10 {
                let params = optimizer.random_suggest(&mut rng);
                let trial = TrialResult {
                    trial_number: i,
                    hyperparameters: Hyperparameters {
                        quality_score: 0.5 + i as f64 * 0.1,
                        ..params
                    },
                    quality_score: 0.5 + i as f64 * 0.1,
                    avg_reward: 0.0,
                    duration_seconds: 100.0,
                };
                optimizer.tell(trial);
            }

            assert_eq!(optimizer.num_trials(), 10);
        }
    }
}
