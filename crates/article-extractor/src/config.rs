use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;

/// Configuration for the article extractor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Paths
    pub model_path: Option<PathBuf>,
    pub site_profiles_dir: PathBuf,
    pub output_dir: PathBuf,
    pub models_dir: PathBuf,
    pub use_cpu_for_tuning: bool,

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

    // NEW: Performance tuning
    pub min_replay_size: usize,          // Start training after this many experiences
    pub train_freq: usize,                // Train every N steps
    pub num_train_steps_per_episode: usize, // Multiple gradient updates per episode
    pub max_html_samples: usize,          // Limit dataset size
    pub sample_batch_load_size: usize,    // Load HTML in batches
    pub prefetch_samples: bool,           // Async sample loading

    // NEW: Metrics
    pub metrics_window: usize,            // Window for moving averages
    pub checkpoint_freq: usize,           // Save every N episodes
    pub log_freq: usize,                  // Progress bar update frequency

    // State/Action space
    pub state_dim: usize,
    pub num_discrete_actions: usize,
    pub num_continuous_params: usize,
    pub num_candidate_nodes: usize,

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

            num_episodes: 10000,

            // OPTIMIZED: Smaller batch size for better GPU utilization
            batch_size: 512,  // Down from 8192 - better for gradient updates

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

            stopwords: Self::default_stopwords(),
        }
    }
}

impl Config {
    /// Create high-performance GPU config
    pub fn gpu_optimized() -> Self {
        let mut config = Self::default();

        // GPU-specific optimizations
        config.batch_size = 1024;                    // Larger but not excessive
        config.num_train_steps_per_episode = 8;     // More gradient updates
        config.train_freq = 2;                       // Train more frequently
        config.max_html_samples = 3000;              // Even smaller dataset
        config.metrics_window = 25;                  // Faster feedback

        config
    }

    /// Create config optimized for your specific GPU (7.6GB)
    pub fn rtx_3080_optimized() -> Self {
        let mut config = Self::default();

        // Maximize GPU utilization for RTX 3080 Ti / similar
        config.batch_size = 2048;                    // Use more GPU memory
        config.num_train_steps_per_episode = 16;    // Many gradient updates per episode
        config.train_freq = 1;                       // Train every step
        config.replay_buffer_size = 200000;          // Larger buffer
        config.min_replay_size = 10000;              // More initial experiences
        config.max_html_samples = 2000;              // Smaller, diverse dataset
        config.metrics_window = 25;
        config.checkpoint_freq = 250;
        config.target_update_freq = 250;

        config
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
