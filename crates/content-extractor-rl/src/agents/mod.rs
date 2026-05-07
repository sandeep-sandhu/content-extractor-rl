// ============================================================================
// FILE: crates/content-extractor-rl/src/agents/mod.rs
// ============================================================================
pub mod dqn_agent;
pub mod ppo_agent;
pub mod sac_agent;

use crate::{Result, replay_buffer::PrioritizedReplayBuffer};
use candle_core::Device;
use std::path::Path;
use candle_nn::VarMap;
use serde::{Serialize, Deserialize};
use crate::models::NetworkConfig;

/// Algorithm type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum AlgorithmType {
    #[default]
    DuelingDQN,
    PPO,
    SAC,
    TD3,
    Rainbow,
}
impl std::str::FromStr for AlgorithmType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "dqn" | "dueling_dqn" | "duelingdqn" => Ok(AlgorithmType::DuelingDQN),
            "ppo" => Ok(AlgorithmType::PPO),
            "sac" => Ok(AlgorithmType::SAC),
            "td3" => Ok(AlgorithmType::TD3),
            "rainbow" => Ok(AlgorithmType::Rainbow),
            _ => Err(format!("Unknown algorithm type: {}. Supported: dqn, ppo, sac, td3, rainbow", s))
        }
    }
}
impl std::fmt::Display for AlgorithmType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgorithmType::DuelingDQN => write!(f, "DuelingDQN"),
            AlgorithmType::PPO => write!(f, "PPO"),
            AlgorithmType::SAC => write!(f, "SAC"),
            AlgorithmType::TD3 => write!(f, "TD3"),
            AlgorithmType::Rainbow => write!(f, "Rainbow"),
        }
    }
}

/// Common trait for all RL agents
pub trait RLAgent: Send + Sync {
    /// Select action given state and exploration parameter
    /// Returns: (discrete_action, continuous_params, optional_log_prob)
    fn select_action(&self, state: &[f32], epsilon: f32) -> Result<(usize, Vec<f32>)>;

    /// Save model with metadata
    fn save_with_metadata(&self, path: &Path, training_episodes: usize, hyperparameters: std::collections::HashMap<String, f64>) -> Result<()>;

    /// Save model to disk (uses default metadata)
    fn save(&self, path: &Path) -> Result<()>;

    /// Train on a batch of experiences
    /// Returns: loss value
    fn train_step(&mut self, replay_buffer: &mut PrioritizedReplayBuffer, batch_size: usize) -> Result<f32>;

    /// Update target network (if applicable, no-op for on-policy methods)
    fn update_target_network(&mut self);

    /// Get training step count
    fn get_step_count(&self) -> usize;

    /// Get algorithm type
    fn algorithm_type(&self) -> AlgorithmType;

    /// Get algorithm-specific info for logging
    fn get_info(&self) -> AgentInfo;

}

/// Agent information for logging and tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub algorithm: AlgorithmType,
    pub num_parameters: usize,
    pub state_dim: usize,
    pub num_actions: usize,
    pub continuous_params: usize,
    pub version: String,
    pub features: Vec<String>,
}
/// Factory for creating RL agents
pub struct AgentFactory;

impl AgentFactory {
    /// Create agent from configuration
    pub fn create(
        algorithm: AlgorithmType,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        gamma: f32,
        lr: f64,
        device: &Device,
    ) -> Result<Box<dyn RLAgent>> {
        match algorithm {
            AlgorithmType::DuelingDQN => {
                let network_config = NetworkConfig {
                    state_dim,
                    num_actions,
                    num_params,
                    hidden_layers: vec![512, 256, 128],
                    use_layer_norm: true,
                    dropout: 0.1,
                    value_hidden: 64,
                    advantage_hidden: 64,
                };

                // Create varmap for this agent
                let varmap = VarMap::new();

                let agent = dqn_agent::DQNAgent::new(
                    network_config, gamma, lr, device, varmap
                )?;
                Ok(Box::new(agent))
            }
            AlgorithmType::PPO => {
                let varmap = candle_nn::VarMap::new();
                let agent = ppo_agent::PPOAgent::new(
                    state_dim, num_actions, num_params, gamma, lr, device, varmap
                )?;
                Ok(Box::new(agent))
            }
            AlgorithmType::SAC => {
                let actor_varmap = candle_nn::VarMap::new();
                let critic_varmap = candle_nn::VarMap::new();
                let agent = sac_agent::SACAgent::new(
                    state_dim, num_actions, num_params, gamma, lr, device,
                    actor_varmap, critic_varmap
                )?;
                Ok(Box::new(agent))
            }
            _ => Err(crate::ExtractionError::ModelError(
                format!("Algorithm {} not yet implemented", algorithm)
            ))
        }
    }

    /// Load agent from saved model
    pub fn load(
        path: &Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> Result<Box<dyn RLAgent>> {
        let algorithm = Self::detect_algorithm(path)?;

        match algorithm {
            AlgorithmType::DuelingDQN => {
                let agent = dqn_agent::DQNAgent::load_with_device(
                    path, state_dim, num_actions, num_params, device
                )?;
                Ok(Box::new(agent))
            }
            AlgorithmType::PPO => {
                let agent = ppo_agent::PPOAgent::load_with_device(
                    path, state_dim, num_actions, num_params, device
                )?;
                Ok(Box::new(agent))
            }
            AlgorithmType::SAC => {
                let agent = sac_agent::SACAgent::load_with_device(
                    path, state_dim, num_actions, num_params, device
                )?;
                Ok(Box::new(agent))
            }
            _ => Err(crate::ExtractionError::ModelError(
                format!("Algorithm {} loading not implemented", algorithm)
            ))
        }
    }

    /// Detect algorithm type from saved model
    fn detect_algorithm(path: &Path) -> Result<AlgorithmType> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)?;
        let mut metadata_len_bytes = [0u8; 8];
        file.read_exact(&mut metadata_len_bytes)?;
        let metadata_len = u64::from_le_bytes(metadata_len_bytes) as usize;

        let mut metadata_bytes = vec![0u8; metadata_len];
        file.read_exact(&mut metadata_bytes)?;

        let metadata_json = String::from_utf8(metadata_bytes)
            .map_err(|e| crate::ExtractionError::ParseError(e.to_string()))?;

        #[derive(Deserialize)]
        struct Metadata {
            architecture: String,
        }

        let metadata: Metadata = serde_json::from_str(&metadata_json)
            .map_err(|e| crate::ExtractionError::ParseError(e.to_string()))?;

        metadata.architecture.parse()
            .map_err(|e: String| crate::ExtractionError::ParseError(e))
    }
    
}