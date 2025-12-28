//! Soft Actor-Critic network
// ============================================================================
// FILE: crates/article-extractor/src/agents/sac_agent.rs
// ============================================================================

use candle_core::{Device, Tensor, DType, Var};
use candle_nn::{VarBuilder, Optimizer, AdamW, ParamsAdamW, VarMap, Linear, Module, linear, layer_norm, LayerNorm};
use crate::replay_buffer::PrioritizedReplayBuffer;
use crate::{Result, agents::{RLAgent, AlgorithmType, AgentInfo}};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use tracing::{info, warn};
use std::path::Path;
use std::collections::HashMap;
use crate::models::ModelMetadata;
use candle_nn::ops::softmax;

/// Actor network for SAC (outputs mean and log_std)
pub struct SACActorNetwork {
    fc1: Linear,
    ln1: LayerNorm,
    fc2: Linear,
    ln2: LayerNorm,
    fc3: Linear,
    ln3: LayerNorm,
    // Discrete action head
    action_logits: Linear,

    // Continuous parameter heads
    param_mean: Linear,
    param_logstd: Linear,

    device: Device,
    num_actions: usize,
    num_params: usize,
}
impl SACActorNetwork {
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        vb: VarBuilder,
    ) -> candle_core::error::Result<Self> {
        let device = vb.device().clone();
        let fc1 = linear(state_dim, 512, vb.pp("fc1"))?;
        let ln1 = layer_norm(512, 1e-5, vb.pp("ln1"))?;
        let fc2 = linear(512, 256, vb.pp("fc2"))?;
        let ln2 = layer_norm(256, 1e-5, vb.pp("ln2"))?;
        let fc3 = linear(256, 128, vb.pp("fc3"))?;
        let ln3 = layer_norm(128, 1e-5, vb.pp("ln3"))?;

        let action_logits = linear(128, num_actions, vb.pp("action_logits"))?;
        let param_mean = linear(128, num_params, vb.pp("param_mean"))?;
        let param_logstd = linear(128, num_params, vb.pp("param_logstd"))?;

        Ok(Self {
            fc1, ln1, fc2, ln2, fc3, ln3,
            action_logits,
            param_mean,
            param_logstd,
            device,
            num_actions,
            num_params,
        })
    }

    pub fn forward(&self, state: &Tensor) -> candle_core::error::Result<(Tensor, Tensor, Tensor)> {
        let mut x = self.fc1.forward(state)?;
        x = self.ln1.forward(&x)?;
        x = x.relu()?;

        x = self.fc2.forward(&x)?;
        x = self.ln2.forward(&x)?;
        x = x.relu()?;

        x = self.fc3.forward(&x)?;
        x = self.ln3.forward(&x)?;
        let features = x.relu()?;

        let action_logits = self.action_logits.forward(&features)?;
        let param_mean = self.param_mean.forward(&features)?.tanh()?;
        let param_logstd = self.param_logstd.forward(&features)?.clamp(-20.0, 2.0)?;

        Ok((action_logits, param_mean, param_logstd))
    }
}
/// Twin Q-network for SAC
pub struct SACCriticNetwork {
    // Q1 network
    q1_fc1: Linear,
    q1_ln1: LayerNorm,
    q1_fc2: Linear,
    q1_ln2: LayerNorm,
    q1_output: Linear,
    // Q2 network (twin)
    q2_fc1: Linear,
    q2_ln1: LayerNorm,
    q2_fc2: Linear,
    q2_ln2: LayerNorm,
    q2_output: Linear,

    num_actions: usize,
    num_params: usize,
}
impl SACCriticNetwork {
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        vb: VarBuilder,
    ) -> candle_core::error::Result<Self> {
        // Combined state-action dimension
        let input_dim = state_dim + num_actions + num_params;
        // Q1 network
        let q1_fc1 = linear(input_dim, 512, vb.pp("q1_fc1"))?;
        let q1_ln1 = layer_norm(512, 1e-5, vb.pp("q1_ln1"))?;
        let q1_fc2 = linear(512, 256, vb.pp("q1_fc2"))?;
        let q1_ln2 = layer_norm(256, 1e-5, vb.pp("q1_ln2"))?;
        let q1_output = linear(256, 1, vb.pp("q1_output"))?;

        // Q2 network
        let q2_fc1 = linear(input_dim, 512, vb.pp("q2_fc1"))?;
        let q2_ln1 = layer_norm(512, 1e-5, vb.pp("q2_ln1"))?;
        let q2_fc2 = linear(512, 256, vb.pp("q2_fc2"))?;
        let q2_ln2 = layer_norm(256, 1e-5, vb.pp("q2_ln2"))?;
        let q2_output = linear(256, 1, vb.pp("q2_output"))?;

        Ok(Self {
            q1_fc1, q1_ln1, q1_fc2, q1_ln2, q1_output,
            q2_fc1, q2_ln1, q2_fc2, q2_ln2, q2_output,
            num_actions,
            num_params,
        })
    }

    pub fn forward(
        &self,
        state: &Tensor,
        action_discrete: &Tensor,
        action_continuous: &Tensor,
    ) -> candle_core::error::Result<(Tensor, Tensor)> {
        // Concatenate state and actions
        let state_action = Tensor::cat(&[state, action_discrete, action_continuous], 1)?;

        // Q1 forward
        let mut x1 = self.q1_fc1.forward(&state_action)?;
        x1 = self.q1_ln1.forward(&x1)?;
        x1 = x1.relu()?;
        x1 = self.q1_fc2.forward(&x1)?;
        x1 = self.q1_ln2.forward(&x1)?;
        x1 = x1.relu()?;
        let q1 = self.q1_output.forward(&x1)?.squeeze(1)?;

        // Q2 forward
        let mut x2 = self.q2_fc1.forward(&state_action)?;
        x2 = self.q2_ln1.forward(&x2)?;
        x2 = x2.relu()?;
        x2 = self.q2_fc2.forward(&x2)?;
        x2 = self.q2_ln2.forward(&x2)?;
        x2 = x2.relu()?;
        let q2 = self.q2_output.forward(&x2)?.squeeze(1)?;

        Ok((q1, q2))
    }
}

/// SAC Agent with automatic entropy tuning
pub struct SACAgent {
    actor: SACActorNetwork,
    critic: SACCriticNetwork,
    target_critic: SACCriticNetwork,
    actor_optimizer: AdamW,
    critic_optimizer: AdamW,

    // Automatic temperature tuning
    log_alpha: Var,
    alpha_optimizer: AdamW,
    target_entropy: f32,

    actor_varmap: VarMap,
    critic_varmap: VarMap,
    alpha_varmap: VarMap,

    num_actions: usize,
    num_params: usize,
    gamma: f32,
    tau: f32,  // Soft update coefficient
    step_count: usize,
    device: Device,
}


fn save_linear_helper(
    tensors: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
    name: &str,
    linear: &Linear
) -> Result<()> {
    let weight = linear.weight();
    let weight_shape = weight.dims().to_vec();
    let weight_data = weight.flatten_all()
        .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?
        .to_vec1::<f32>()
        .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
    tensors.insert(format!("{}.weight", name), (weight_shape, weight_data));

    if let Some(bias) = linear.bias() {
        let bias_shape = bias.dims().to_vec();
        let bias_data = bias.flatten_all()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        tensors.insert(format!("{}.bias", name), (bias_shape, bias_data));
    }
    Ok(())
}

fn save_layernorm_helper(
    tensors: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
    name: &str,
    ln: &LayerNorm
) -> Result<()> {
    let weight = ln.weight();
    let weight_shape = weight.dims().to_vec();
    let weight_data = weight.flatten_all()
        .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?
        .to_vec1::<f32>()
        .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
    tensors.insert(format!("{}.weight", name), (weight_shape, weight_data));

    if let Some(bias) = ln.bias() {
        let bias_shape = bias.dims().to_vec();
        let bias_data = bias.flatten_all()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        tensors.insert(format!("{}.bias", name), (bias_shape, bias_data));
    }
    Ok(())
}

impl SACAgent {
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        gamma: f32,
        lr: f64,
        device: &Device,
        actor_varmap: VarMap,
        critic_varmap: VarMap,
    ) -> Result<Self> {
        // Create actor
        let actor_vb = VarBuilder::from_varmap(&actor_varmap, DType::F32, device);
        let actor = SACActorNetwork::new(state_dim, num_actions, num_params, actor_vb)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        // Create critic and target critic
        let critic_vb = VarBuilder::from_varmap(&critic_varmap, DType::F32, device);
        let critic = SACCriticNetwork::new(state_dim, num_actions, num_params, critic_vb.pp("online"))
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let target_critic_varmap = VarMap::new();
        let target_vb = VarBuilder::from_varmap(&target_critic_varmap, DType::F32, device);
        let target_critic = SACCriticNetwork::new(state_dim, num_actions, num_params, target_vb.pp("target"))
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Initialize temperature (alpha) for entropy regularization
        let alpha_varmap = VarMap::new();
        let log_alpha_init = Tensor::new(&[0.0f32], device)?;
        let log_alpha = Var::from_tensor(&log_alpha_init)?;

        // Target entropy: -dim(action_space)
        let target_entropy = -(num_actions as f32 + num_params as f32);

        // Create optimizers
        let actor_params = ParamsAdamW { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 };
        let actor_optimizer = AdamW::new(actor_varmap.all_vars(), actor_params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let critic_params = ParamsAdamW { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 };
        let critic_optimizer = AdamW::new(critic_varmap.all_vars(), critic_params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let alpha_params = ParamsAdamW { lr: lr * 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0 };
        let alpha_optimizer = AdamW::new(vec![log_alpha.clone()], alpha_params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(Self {
            actor,
            critic,
            target_critic,
            actor_optimizer,
            critic_optimizer,
            log_alpha,
            alpha_optimizer,
            target_entropy,
            actor_varmap,
            critic_varmap,
            alpha_varmap,
            num_actions,
            num_params,
            gamma,
            tau: 0.005,  // Soft update coefficient
            step_count: 0,
            device: device.clone(),
        })
    }

    /// Sample action from policy
    fn sample_action(&self, state: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (action_logits, param_mean, param_logstd) = self.actor.forward(state)?;

        // Sample discrete action (Gumbel-Softmax for differentiability)
        let action_probs = softmax(&action_logits, 1)?;
        let action_discrete_onehot = self.gumbel_softmax(&action_logits, 1.0)?;

        // Sample continuous params (reparameterization trick)
        let param_std = param_logstd.exp()?;

        // FIXED: Use randn with shape instead of randn_like
        let noise = Tensor::randn(0.0, 1.0, param_mean.shape(), &self.device)?;
        let action_continuous = (param_mean.clone() + param_std.clone() * noise)?;

        // Calculate log probability for entropy
        let log_prob_discrete = action_probs.log()?.mul(&action_discrete_onehot)?.sum(1)?;
        let log_prob_continuous = self.gaussian_log_prob(&param_mean, &param_std, &action_continuous)?;
        let log_prob = (log_prob_discrete + log_prob_continuous)?;

        Ok((action_discrete_onehot, action_continuous, log_prob))
    }

    /// Gumbel-Softmax for discrete actions
    fn gumbel_softmax(&self, logits: &Tensor, temperature: f32) -> candle_core::error::Result<Tensor> {
        // FIXED: Use randn with shape and manually compute gumbel
        let uniform = Tensor::rand(0.0, 1.0, logits.shape(), logits.device())?;
        let gumbel = (uniform.log()?.neg()?).log()?.neg()?;

        let temp_tensor = Tensor::new(&[temperature], logits.device())?;
        let y = (logits.clone() + gumbel)?.div(&temp_tensor)?;
        softmax(&y, 1)
    }

    /// Gaussian log probability
    fn gaussian_log_prob(&self, mean: &Tensor, std: &Tensor, value: &Tensor) -> candle_core::error::Result<Tensor> {
        let variance = std.sqr()?;
        let log_std = std.log()?;
        let diff = (value - mean)?;

        let log_prob = -0.5 * (
            diff.sqr()?.div(&variance)? +
                Tensor::new(&[2.0 * std::f32::consts::PI], mean.device())?.log()? +
                log_std.mul(&Tensor::new(&[2.0f32], mean.device())?)?
        )?;

        log_prob?.sum(1)
    }

    /// Soft update of target network
    fn soft_update_target(&mut self) -> Result<()> {
        // This is a placeholder - in practice, you'd copy weights with soft update
        // target = tau * online + (1 - tau) * target
        warn!("SAC soft target update not yet implemented");
        Ok(())
    }

    /// Save SAC model to file with metadata
    pub fn save_to_file(&self, path: &Path, metadata: ModelMetadata) -> Result<()> {
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create(path)?;

        // Write metadata
        let metadata_json = serde_json::to_string(&metadata)
            .map_err(|e| crate::ExtractionError::ParseError(e.to_string()))?;
        let metadata_bytes = metadata_json.as_bytes();
        let metadata_len = metadata_bytes.len() as u64;

        file.write_all(&metadata_len.to_le_bytes())?;
        file.write_all(metadata_bytes)?;

        // Collect all tensors - FIXED: Use helper functions
        let mut tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();

        // Save actor network
        save_linear_helper(&mut tensors, "actor.fc1", &self.actor.fc1)?;
        save_layernorm_helper(&mut tensors, "actor.ln1", &self.actor.ln1)?;
        save_linear_helper(&mut tensors, "actor.fc2", &self.actor.fc2)?;
        save_layernorm_helper(&mut tensors, "actor.ln2", &self.actor.ln2)?;
        save_linear_helper(&mut tensors, "actor.fc3", &self.actor.fc3)?;
        save_layernorm_helper(&mut tensors, "actor.ln3", &self.actor.ln3)?;
        save_linear_helper(&mut tensors, "actor.action_logits", &self.actor.action_logits)?;
        save_linear_helper(&mut tensors, "actor.param_mean", &self.actor.param_mean)?;
        save_linear_helper(&mut tensors, "actor.param_logstd", &self.actor.param_logstd)?;

        // Save critic network (Q1 and Q2)
        save_linear_helper(&mut tensors, "critic.q1_fc1", &self.critic.q1_fc1)?;
        save_layernorm_helper(&mut tensors, "critic.q1_ln1", &self.critic.q1_ln1)?;
        save_linear_helper(&mut tensors, "critic.q1_fc2", &self.critic.q1_fc2)?;
        save_layernorm_helper(&mut tensors, "critic.q1_ln2", &self.critic.q1_ln2)?;
        save_linear_helper(&mut tensors, "critic.q1_output", &self.critic.q1_output)?;

        save_linear_helper(&mut tensors, "critic.q2_fc1", &self.critic.q2_fc1)?;
        save_layernorm_helper(&mut tensors, "critic.q2_ln1", &self.critic.q2_ln1)?;
        save_linear_helper(&mut tensors, "critic.q2_fc2", &self.critic.q2_fc2)?;
        save_layernorm_helper(&mut tensors, "critic.q2_ln2", &self.critic.q2_ln2)?;
        save_linear_helper(&mut tensors, "critic.q2_output", &self.critic.q2_output)?;

        // Save log_alpha (temperature parameter)
        let log_alpha_tensor = self.log_alpha.as_tensor();
        let log_alpha_shape = log_alpha_tensor.dims().to_vec();
        let log_alpha_data = log_alpha_tensor.flatten_all()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        tensors.insert("log_alpha".to_string(), (log_alpha_shape, log_alpha_data));

        // Write tensor count
        let tensor_count = tensors.len() as u64;
        file.write_all(&tensor_count.to_le_bytes())?;

        // Write each tensor
        for (name, (shape, data)) in tensors.iter() {
            let name_bytes = name.as_bytes();
            let name_len = name_bytes.len() as u64;
            file.write_all(&name_len.to_le_bytes())?;
            file.write_all(name_bytes)?;

            let shape_len = shape.len() as u64;
            file.write_all(&shape_len.to_le_bytes())?;
            for &dim in shape {
                file.write_all(&(dim as u64).to_le_bytes())?;
            }

            let data_len = data.len() as u64;
            file.write_all(&data_len.to_le_bytes())?;
            for &value in data {
                file.write_all(&value.to_le_bytes())?;
            }
        }

        let file_size = std::fs::metadata(path)?.len();
        tracing::info!("SAC model saved: {} bytes", file_size);

        Ok(())
    }

    /// Load SAC model from file
    pub fn load_from_file(
        path: &Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> Result<Self> {
        use std::fs::File;
        use std::io::Read;

        tracing::info!("Loading SAC model from: {}", path.display());

        let mut file = File::open(path)?;

        // Read metadata
        let mut metadata_len_bytes = [0u8; 8];
        file.read_exact(&mut metadata_len_bytes)?;
        let metadata_len = u64::from_le_bytes(metadata_len_bytes) as usize;

        let mut metadata_bytes = vec![0u8; metadata_len];
        file.read_exact(&mut metadata_bytes)?;

        let metadata_json = String::from_utf8(metadata_bytes)
            .map_err(|e| crate::ExtractionError::ParseError(e.to_string()))?;
        let _metadata: ModelMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| crate::ExtractionError::ParseError(e.to_string()))?;

        tracing::info!("Model metadata loaded, loading tensors...");

        // Read tensor count
        let mut tensor_count_bytes = [0u8; 8];
        file.read_exact(&mut tensor_count_bytes)?;
        let tensor_count = u64::from_le_bytes(tensor_count_bytes) as usize;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        for _ in 0..tensor_count {
            let mut name_len_bytes = [0u8; 8];
            file.read_exact(&mut name_len_bytes)?;
            let name_len = u64::from_le_bytes(name_len_bytes) as usize;

            let mut name_bytes = vec![0u8; name_len];
            file.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| crate::ExtractionError::ParseError(e.to_string()))?;

            let mut shape_len_bytes = [0u8; 8];
            file.read_exact(&mut shape_len_bytes)?;
            let shape_len = u64::from_le_bytes(shape_len_bytes) as usize;

            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let mut dim_bytes = [0u8; 8];
                file.read_exact(&mut dim_bytes)?;
                shape.push(u64::from_le_bytes(dim_bytes) as usize);
            }

            let mut data_len_bytes = [0u8; 8];
            file.read_exact(&mut data_len_bytes)?;
            let data_len = u64::from_le_bytes(data_len_bytes) as usize;

            let mut data = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                let mut value_bytes = [0u8; 4];
                file.read_exact(&mut value_bytes)?;
                data.push(f32::from_le_bytes(value_bytes));
            }

            let tensor = Tensor::from_vec(data, shape.as_slice(), device)
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
            tensors.insert(name, tensor);
        }

        tracing::info!("Loaded {} tensors, reconstructing model...", tensors.len());

        // Make varmaps mutable
        let mut actor_varmap = VarMap::new();
        let mut critic_varmap = VarMap::new();

        for (name, tensor) in tensors.iter() {
            let var = Var::from_tensor(tensor)
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

            if name.starts_with("actor.") {
                let actor_name = name.strip_prefix("actor.").unwrap();
                actor_varmap.set_one(actor_name, var.as_tensor())
                    .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
            } else if name.starts_with("critic.") {
                let critic_name = name.strip_prefix("critic.").unwrap();
                critic_varmap.set_one(critic_name, var.as_tensor())
                    .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
            }
        }

        Self::new(state_dim, num_actions, num_params, 0.95, 3e-4, device, actor_varmap, critic_varmap)
    }

    /// Update load_with_device to use load_from_file
    pub fn load_with_device(
        path: &Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::load_from_file(path, state_dim, num_actions, num_params, device)
    }
}

impl RLAgent for SACAgent {
    fn select_action(&self, state: &[f32], _epsilon: f32) -> Result<(usize, Vec<f32>)> {
        let state_tensor = Tensor::from_vec(state.to_vec(), &[1, state.len()], &self.device)?;
        let (action_logits, param_mean, _param_logstd) = self.actor.forward(&state_tensor)?;

        // For inference, use mean of distributions
        let action_probs = softmax(&action_logits, 1)?.to_vec2::<f32>()?;
        let discrete_action = action_probs[0].iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let continuous_params = param_mean.to_vec2::<f32>()?[0].clone();

        Ok((discrete_action, continuous_params))
    }

    fn train_step(&mut self, replay_buffer: &mut PrioritizedReplayBuffer, batch_size: usize) -> Result<f32> {
        let batch = replay_buffer.sample(batch_size);
        if batch.is_none() {
            return Ok(0.0);
        }

        let batch = batch.unwrap();
        let experiences = &batch.experiences;

        if experiences.is_empty() {
            return Ok(0.0);
        }

        // Convert to tensors
        let state_dim = experiences[0].state.len();
        let states_flat: Vec<f32> = experiences.iter().flat_map(|e| e.state.clone()).collect();
        let states = Tensor::from_vec(states_flat, &[experiences.len(), state_dim], &self.device)?;

        let next_states_flat: Vec<f32> = experiences.iter().flat_map(|e| e.next_state.clone()).collect();
        let next_states = Tensor::from_vec(next_states_flat, &[experiences.len(), state_dim], &self.device)?;

        let rewards: Vec<f32> = experiences.iter().map(|e| e.reward).collect();
        let rewards_tensor = Tensor::from_vec(rewards, &[experiences.len()], &self.device)?;

        let dones: Vec<f32> = experiences.iter().map(|e| if e.done { 1.0 } else { 0.0 }).collect();
        let dones_tensor = Tensor::from_vec(dones, &[experiences.len()], &self.device)?;

        // Get current alpha (temperature)
        let alpha = self.log_alpha.as_tensor().exp()?;

        // Update critic
        let (next_action_discrete, next_action_continuous, next_log_prob) = self.sample_action(&next_states)?;
        let (next_q1, next_q2) = self.target_critic.forward(&next_states, &next_action_discrete, &next_action_continuous)?;
        let next_q = next_q1.minimum(&next_q2)?;

        let alpha_broadcast = alpha.broadcast_as(next_log_prob.shape())?;
        let target_q = (
            rewards_tensor.clone() +
                (Tensor::ones_like(&dones_tensor)? - dones_tensor.clone())?.mul(&Tensor::new(&[self.gamma], &self.device)?)?.mul(
                    &(next_q.clone() - alpha_broadcast.mul(&next_log_prob)?)?
                )?
        )?;

        // Current actions (from experience)
        let actions_discrete: Vec<f32> = experiences.iter()
            .flat_map(|e| {
                let mut onehot = vec![0.0f32; self.num_actions];
                onehot[e.action.0] = 1.0;
                onehot
            })
            .collect();
        let actions_discrete_tensor = Tensor::from_vec(actions_discrete, &[experiences.len(), self.num_actions], &self.device)?;

        let actions_continuous_flat: Vec<f32> = experiences.iter().flat_map(|e| e.action.1.clone()).collect();
        let actions_continuous_tensor = Tensor::from_vec(actions_continuous_flat, &[experiences.len(), self.num_params], &self.device)?;

        let (current_q1, current_q2) = self.critic.forward(&states, &actions_discrete_tensor, &actions_continuous_tensor)?;

        let critic_loss = (
            (current_q1.clone() - target_q.clone())?.sqr()? +
                (current_q2 - target_q)?.sqr()?
        )?.mean_all()?;

        // Backward and update critic
        let critic_grads = critic_loss.backward()?;
        self.critic_optimizer.step(&critic_grads)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Update actor
        let (sampled_action_discrete, sampled_action_continuous, log_prob) = self.sample_action(&states)?;
        let (q1_new, q2_new) = self.critic.forward(&states, &sampled_action_discrete, &sampled_action_continuous)?;
        let q_new = q1_new.minimum(&q2_new)?;

        let alpha_broadcast = alpha.broadcast_as(log_prob.shape())?;
        let actor_loss = (alpha_broadcast.mul(&log_prob)? - q_new)?.mean_all()?;

        let actor_grads = actor_loss.backward()?;
        self.actor_optimizer.step(&actor_grads)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Update temperature (alpha)
        let alpha_loss = (self.log_alpha.as_tensor().neg()? *
            (log_prob.clone() + Tensor::new(&[self.target_entropy], &self.device)?)?.detach())?
            .mean_all()?;

        let alpha_grads = alpha_loss.backward()?;
        self.alpha_optimizer.step(&alpha_grads)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Soft update target network
        self.soft_update_target()?;

        self.step_count += 1;

        Ok(critic_loss.to_scalar::<f32>()?)
    }

    fn update_target_network(&mut self) {
        // SAC uses soft updates, called in train_step
    }

    fn get_step_count(&self) -> usize {
        self.step_count
    }

    fn save_with_metadata(
        &self,
        path: &Path,
        training_episodes: usize,
        hyperparameters: HashMap<String, f64>,
    ) -> Result<()> {
        let metadata = ModelMetadata::new(
            300,  // state_dim
            self.num_actions,
            self.num_params,
            AlgorithmType::SAC,
            training_episodes,
            hyperparameters,
        );

        self.save_to_file(path, metadata)
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.save_with_metadata(path, 0, HashMap::new())
    }

    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::SAC
    }

    fn get_info(&self) -> AgentInfo {
        AgentInfo {
            algorithm: AlgorithmType::SAC,
            num_parameters: 0,
            state_dim: 0,
            num_actions: self.num_actions,
            continuous_params: self.num_params,
            version: "1.0.0".to_string(),
            features: vec![
                "twin_q".to_string(),
                "entropy_regularization".to_string(),
                "automatic_temperature".to_string(),
                "off_policy".to_string(),
            ],
        }
    }

}
