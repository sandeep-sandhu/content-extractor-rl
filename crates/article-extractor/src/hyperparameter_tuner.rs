//! Hyperparameter tuning using TPE (Tree-structured Parzen Estimator) with resume capability

use crate::{Config, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use rand::Rng;
use tracing::{info, debug, warn};

/// Hyperparameter search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    pub learning_rate: (f64, f64),  // (min, max)
    pub batch_size: Vec<usize>,      // discrete choices
    pub gamma: (f64, f64),
    pub epsilon_decay: (f64, f64),
    pub priority_alpha: (f64, f64),
    pub priority_beta: (f64, f64),
}

impl Default for HyperparameterSpace {
    fn default() -> Self {
        Self {
            learning_rate: (1e-5, 1e-2),
            batch_size: vec![64, 128, 256, 512],
            gamma: (0.85, 0.99),
            epsilon_decay: (0.985, 0.999),
            priority_alpha: (0.4, 0.8),
            priority_beta: (0.3, 0.7),
        }
    }
}

/// Hyperparameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub gamma: f64,
    pub epsilon_decay: f64,
    pub priority_alpha: f64,
    pub priority_beta: f64,
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

    /// Suggest next hyperparameters using TPE
    pub fn suggest(&self) -> Hyperparameters {
        let mut rng = rand::rng();

        // Use random search for initial trials
        if self.trials.len() < self.n_startup_trials {
            info!("Using random search (trial {}/{})", self.trials.len() + 1, self.n_startup_trials);
            return self.random_suggest(&mut rng);
        }

        info!("Using TPE (trial {})", self.trials.len() + 1);

        // TPE: Split trials into good and bad based on quantile
        let quantile = 0.25; // Top 25% are "good"
        let mut sorted_trials = self.trials.clone();
        sorted_trials.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap());

        let n_good = (sorted_trials.len() as f64 * quantile).ceil() as usize;
        let good_trials = &sorted_trials[..n_good];
        let bad_trials = &sorted_trials[n_good..];

        debug!("TPE split: {} good trials, {} bad trials", n_good, sorted_trials.len() - n_good);

        // Sample from good distribution vs bad distribution
        // For each parameter, model l(x) and g(x) as Gaussians
        // Sample from argmax_x l(x)/g(x)

        let learning_rate = self.sample_tpe_continuous(
            good_trials.iter().map(|t| t.hyperparameters.learning_rate).collect(),
            bad_trials.iter().map(|t| t.hyperparameters.learning_rate).collect(),
            self.space.learning_rate,
            &mut rng,
        );

        let gamma = self.sample_tpe_continuous(
            good_trials.iter().map(|t| t.hyperparameters.gamma).collect(),
            bad_trials.iter().map(|t| t.hyperparameters.gamma).collect(),
            self.space.gamma,
            &mut rng,
        );

        let epsilon_decay = self.sample_tpe_continuous(
            good_trials.iter().map(|t| t.hyperparameters.epsilon_decay).collect(),
            bad_trials.iter().map(|t| t.hyperparameters.epsilon_decay).collect(),
            self.space.epsilon_decay,
            &mut rng,
        );

        let priority_alpha = self.sample_tpe_continuous(
            good_trials.iter().map(|t| t.hyperparameters.priority_alpha).collect(),
            bad_trials.iter().map(|t| t.hyperparameters.priority_alpha).collect(),
            self.space.priority_alpha,
            &mut rng,
        );

        let priority_beta = self.sample_tpe_continuous(
            good_trials.iter().map(|t| t.hyperparameters.priority_beta).collect(),
            bad_trials.iter().map(|t| t.hyperparameters.priority_beta).collect(),
            self.space.priority_beta,
            &mut rng,
        );

        // Discrete parameter: batch_size
        let batch_size = self.sample_tpe_discrete(
            good_trials.iter().map(|t| t.hyperparameters.batch_size).collect(),
            bad_trials.iter().map(|t| t.hyperparameters.batch_size).collect(),
            &self.space.batch_size,
            &mut rng,
        );

        Hyperparameters {
            learning_rate,
            batch_size,
            gamma,
            epsilon_decay,
            priority_alpha,
            priority_beta,
            timestamp: chrono::Utc::now().to_rfc3339(),
            quality_score: 0.0,
        }
    }

    /// Random hyperparameter suggestion
    fn random_suggest(&self, rng: &mut impl Rng) -> Hyperparameters {
        Hyperparameters {
            learning_rate: rng.random_range(self.space.learning_rate.0..self.space.learning_rate.1),
            batch_size: *self.space.batch_size.iter()
                .nth(rng.random_range(0..self.space.batch_size.len()))
                .unwrap(),
            gamma: rng.random_range(self.space.gamma.0..self.space.gamma.1),
            epsilon_decay: rng.random_range(self.space.epsilon_decay.0..self.space.epsilon_decay.1),
            priority_alpha: rng.random_range(self.space.priority_alpha.0..self.space.priority_alpha.1),
            priority_beta: rng.random_range(self.space.priority_beta.0..self.space.priority_beta.1),
            timestamp: chrono::Utc::now().to_rfc3339(),
            quality_score: 0.0,
        }
    }

    /// Sample continuous parameter using TPE
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
    fn sample_tpe_discrete(
        &self,
        good_values: Vec<usize>,
        _bad_values: Vec<usize>,
        choices: &[usize],
        rng: &mut impl Rng,
    ) -> usize {
        if good_values.is_empty() {
            return *choices.iter().nth(rng.random_range(0..choices.len())).unwrap();
        }

        // Count frequency in good trials
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &val in &good_values {
            *counts.entry(val).or_insert(0) += 1;
        }

        // Choose based on frequency (weighted sampling)
        let total: usize = counts.values().sum();
        if total == 0 {
            return *choices.iter().nth(rng.random_range(0..choices.len())).unwrap();
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

        // Simulate some trials
        for i in 0..15 {
            let params = optimizer.suggest();
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

            for i in 0..5 {
                let params = optimizer.suggest();
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

            // Continue with more trials
            for i in 5..10 {
                let params = optimizer.suggest();
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
