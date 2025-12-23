use crate::{Config, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

/// Grid search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSearchConfig {
    pub learning_rates: Vec<f64>,
    pub batch_sizes: Vec<usize>,
    pub gammas: Vec<f64>,
    pub epsilon_decays: Vec<f64>,
    pub priority_alphas: Vec<f64>,
    pub priority_betas: Vec<f64>,
}

impl Default for GridSearchConfig {
    fn default() -> Self {
        Self {
            learning_rates: vec![1e-4, 3e-4, 1e-3, 5e-3, ],
            batch_sizes: vec![256, 512, 1024, 2048, 4096, 8192, 16384],
            gammas: vec![0.90, 0.95, 0.99],
            epsilon_decays: vec![0.990, 0.995, 0.999],
            priority_alphas: vec![0.4, 0.6, 0.7],
            priority_betas: vec![0.4, 0.5, 0.6],
        }
    }
}

/// Hyperparameter search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub params: HashMap<String, f64>,
    pub avg_quality: f32,
    pub avg_reward: f32,
}

/// Hyperparameter search using grid search
pub struct HyperparameterSearch {
    grid_config: GridSearchConfig,
    n_episodes_per_trial: usize,
}

impl HyperparameterSearch {
    /// Create new hyperparameter search
    pub fn new(grid_config: GridSearchConfig, n_episodes_per_trial: usize) -> Self {
        Self {
            grid_config,
            n_episodes_per_trial,
        }
    }

    /// Run grid search
    pub fn run_search(
        &self,
        base_config: &Config,
        html_samples: Vec<(String, String)>,
    ) -> Result<SearchResult> {
        info!("Starting grid search hyperparameter optimization");

        let mut best_result = SearchResult {
            params: HashMap::new(),
            avg_quality: 0.0,
            avg_reward: 0.0,
        };

        let total_combinations = self.grid_config.learning_rates.len()
            * self.grid_config.batch_sizes.len()
            * self.grid_config.gammas.len();

        info!("Total combinations to try: {}", total_combinations);

        let mut trial_count = 0;

        // Grid search over hyperparameters
        for &lr in &self.grid_config.learning_rates {
            for &batch_size in &self.grid_config.batch_sizes {
                for &gamma in &self.grid_config.gammas {
                    for &epsilon_decay in &self.grid_config.epsilon_decays {
                        for &priority_alpha in &self.grid_config.priority_alphas {
                            for &priority_beta in &self.grid_config.priority_betas {
                                trial_count += 1;

                                info!("Trial {}/{}", trial_count, total_combinations);
                                info!("  lr={}, batch={}, gamma={}", lr, batch_size, gamma);

                                // Create config with trial hyperparameters
                                let mut trial_config = base_config.clone();
                                trial_config.learning_rate = lr;
                                trial_config.batch_size = batch_size;
                                trial_config.gamma = gamma;
                                trial_config.epsilon_decay = epsilon_decay;
                                trial_config.priority_alpha = priority_alpha;
                                trial_config.priority_beta = priority_beta;
                                trial_config.num_episodes = self.n_episodes_per_trial;

                                // Run training
                                let (_agent, metrics) = crate::training::train_standard(
                                    &trial_config,
                                    html_samples.clone(),
                                )?;

                                // Calculate average quality
                                let avg_quality = if metrics.episode_qualities.len() >= 100 {
                                    metrics.episode_qualities[metrics.episode_qualities.len() - 100..]
                                        .iter()
                                        .sum::<f32>() / 100.0
                                } else {
                                    metrics.episode_qualities.iter().sum::<f32>()
                                        / metrics.episode_qualities.len().max(1) as f32
                                };

                                let avg_reward = if metrics.episode_rewards.len() >= 100 {
                                    metrics.episode_rewards[metrics.episode_rewards.len() - 100..]
                                        .iter()
                                        .sum::<f32>() / 100.0
                                } else {
                                    metrics.episode_rewards.iter().sum::<f32>()
                                        / metrics.episode_rewards.len().max(1) as f32
                                };

                                info!("  Result: quality={:.4}, reward={:.4}", avg_quality, avg_reward);

                                // Update best result
                                if avg_quality > best_result.avg_quality {
                                    let mut params = HashMap::new();
                                    params.insert("learning_rate".to_string(), lr);
                                    params.insert("batch_size".to_string(), batch_size as f64);
                                    params.insert("gamma".to_string(), gamma);
                                    params.insert("epsilon_decay".to_string(), epsilon_decay);
                                    params.insert("priority_alpha".to_string(), priority_alpha);
                                    params.insert("priority_beta".to_string(), priority_beta);

                                    best_result = SearchResult {
                                        params,
                                        avg_quality,
                                        avg_reward,
                                    };

                                    info!("  ✓ New best result!");
                                }
                            }
                        }
                    }
                }
            }
        }

        info!("Grid search completed!");
        info!("Best hyperparameters:");
        for (key, value) in &best_result.params {
            info!("  {}: {}", key, value);
        }
        info!("Best quality: {:.4}", best_result.avg_quality);

        Ok(best_result)
    }
}
