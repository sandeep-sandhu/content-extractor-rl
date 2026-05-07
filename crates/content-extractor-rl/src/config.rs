// ============================================================================
// FILE: crates/content-extractor-rl/src/config.rs
// ============================================================================

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use crate::agents::AlgorithmType;

/// Configuration for the content extractor rl
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Paths
    pub model_path: Option<PathBuf>,
    pub site_profiles_dir: PathBuf,
    pub output_dir: PathBuf,
    pub models_dir: PathBuf,
    pub use_cpu_for_tuning: bool,

    // Algorithm selection
    #[serde(default)]
    pub algorithm: AlgorithmType,

    // Training hyperparameters
    pub num_episodes: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub gamma: f64,
    pub epsilon_start: f64,
    pub epsilon_end: f64,
    pub epsilon_decay: f64,
    pub target_update_freq: usize,
    pub max_steps_per_episode: usize,

    // Replay buffer
    pub replay_buffer_size: usize,
    pub priority_alpha: f64,
    pub priority_beta: f64,

    // Performance tuning
    pub min_replay_size: usize,
    pub train_freq: usize,
    pub num_train_steps_per_episode: usize,
    pub max_html_samples: usize,
    pub sample_batch_load_size: usize,
    pub prefetch_samples: bool,

    // Metrics
    pub metrics_window: usize,
    pub checkpoint_freq: usize,
    pub log_freq: usize,

    // State/Action space
    pub state_dim: usize,
    pub num_discrete_actions: usize,
    pub num_continuous_params: usize,
    pub num_candidate_nodes: usize,

    // PPO-specific hyperparameters
    pub ppo_clip_epsilon: f32,
    pub ppo_gae_lambda: f32,
    pub ppo_value_loss_coef: f32,
    pub ppo_entropy_coef: f32,
    pub ppo_epochs: usize,

    // Stopwords
    pub stopwords: HashSet<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: std::env::var("ARTICLE_EXTRACTOR_MODEL_PATH")
                .ok()
                .map(PathBuf::from),
            site_profiles_dir: std::env::var("ARTICLE_EXTRACTOR_SITE_PROFILES")
                .ok()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./site_profiles")),
            output_dir: std::env::var("ARTICLE_EXTRACTOR_OUTPUT_DIR")
                .ok()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./output")),
            models_dir: std::env::var("ARTICLE_EXTRACTOR_MODELS_DIR")
                .ok()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("./models")),
            use_cpu_for_tuning: false,

            // Default algorithm
            algorithm: AlgorithmType::DuelingDQN,

            num_episodes: 10000,

            // Smaller batch size for better GPU utilization and is better for gradient updates
            batch_size: 512,

            learning_rate: 3e-4,
            gamma: 0.95,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay: 0.995,

            // OPTIMIZED: More frequent target updates
            target_update_freq: 500,  // Was 1000

            max_steps_per_episode: 20,

            replay_buffer_size: 100000,
            priority_alpha: 0.6,
            priority_beta: 0.4,

            // NEW PERFORMANCE SETTINGS
            min_replay_size: 5000,              // Start training after 5K experiences
            train_freq: 4,                      // Train every 4 steps (more frequent)
            num_train_steps_per_episode: 4,    // 4 gradient updates per episode
            max_html_samples: 5000,             // CRITICAL: Limit to 5K samples
            sample_batch_load_size: 1000,       // Load 1K at a time
            prefetch_samples: true,             // Enable async loading

            // OPTIMIZED METRICS
            metrics_window: 50,                 // Down from 100
            checkpoint_freq: 500,               // More frequent saves
            log_freq: 5,                        // Update progress every 5 episodes

            state_dim: 300,
            num_discrete_actions: 16,
            num_continuous_params: 6,
            num_candidate_nodes: 10,

            // NEW: PPO defaults
            ppo_clip_epsilon: 0.2,
            ppo_gae_lambda: 0.95,
            ppo_value_loss_coef: 0.5,
            ppo_entropy_coef: 0.01,
            ppo_epochs: 4,

            stopwords: Self::default_stopwords(),
        }
    }
}

impl Config {
    /// Create config with specific algorithm
    pub fn with_algorithm(algorithm: AlgorithmType) -> Self {
        Self { algorithm, ..Self::default() }
    }

    /// PPO recommended configuration
    pub fn ppo_recommended() -> Self {
        let mut config = Self::with_algorithm(AlgorithmType::PPO);
        config.batch_size = 2048;
        config.learning_rate = 3e-4;
        config.num_train_steps_per_episode = 8;
        config.ppo_epochs = 10;
        config.ppo_clip_epsilon = 0.2;
        config.ppo_gae_lambda = 0.95;
        config
    }

    /// DQN optimized configuration
    pub fn dqn_optimized() -> Self {
        let mut config = Self::with_algorithm(AlgorithmType::DuelingDQN);
        config.batch_size = 2048;
        config.learning_rate = 0.001;
        config.target_update_freq = 500;
        config
    }

    /// Create high-performance GPU config
    pub fn gpu_optimized() -> Self {
        Self {
            batch_size: 6144,
            num_train_steps_per_episode: 32,
            train_freq: 1,
            replay_buffer_size: 500000,
            min_replay_size: 20000,
            max_html_samples: 10000,
            sample_batch_load_size: 2000,
            learning_rate: 0.00183,
            target_update_freq: 100,
            metrics_window: 50,
            checkpoint_freq: 250,
            ..Self::default()
        }
    }

    /// Create config from environment variables
    pub fn from_env() -> Result<Self> {
        Ok(Self::default())
    }

    /// Setup required directories
    pub fn setup_directories(&self) -> Result<()> {
        std::fs::create_dir_all(&self.site_profiles_dir)?;
        std::fs::create_dir_all(&self.output_dir)?;
        std::fs::create_dir_all(&self.models_dir)?;
        Ok(())
    }

    /// Default English stopwords
    fn default_stopwords() -> HashSet<String> {
        vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "been", "be",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "this", "that", "these", "those",
            "i", "you", "he", "she", "it", "we", "they", "them", "their", "his",
            "her", "its", "our", "your", "who", "what", "where", "when", "why",
            "how", "which", "there", "here", "more", "most", "some", "any", "all",
        ]
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }
}

// Action IDs
pub const ACTION_SELECT_NODE_0: usize = 0;
pub const ACTION_SELECT_NODE_9: usize = 9;
pub const ACTION_SELECT_PARENT: usize = 10;
pub const ACTION_SELECT_SIBLING_LEFT: usize = 11;
pub const ACTION_SELECT_SIBLING_RIGHT: usize = 12;
pub const ACTION_EXPAND_REGION: usize = 13;
pub const ACTION_CONTRACT_REGION: usize = 14;
pub const ACTION_TERMINATE: usize = 15;
