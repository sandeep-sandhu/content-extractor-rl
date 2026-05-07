// ============================================================================
// FILE: crates/content-extractor-rl/src/agents/ppo_agent.rs
// ============================================================================

use candle_core::{Device, Tensor, DType, Var};
use candle_nn::{VarBuilder, Optimizer, AdamW, ParamsAdamW, VarMap, Linear, Module, linear, layer_norm, LayerNorm};
use crate::replay_buffer::{PrioritizedReplayBuffer};
use crate::{Result, agents::{RLAgent, AlgorithmType, AgentInfo}};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::path::{Path, PathBuf};
use crate::models::ModelMetadata;
use std::collections::HashMap;


// Helper functions
fn sample_categorical(probs: &[f32]) -> usize {
    let mut rng = rand::rng();
    let random_val: f32 = rng.random();
    let mut cumsum = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if random_val < cumsum {
            return i;
        }
    }
    probs.len() - 1
}
fn sample_gaussian(means: &[f32], stds: &[f32]) -> Vec<f32> {
    let mut rng = rand::rng();
    means.iter().zip(stds.iter())
        .map(|(&mean, &std)| {
            let normal = Normal::new(mean, std).unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());
            normal.sample(&mut rng)
        })
        .collect()
}

// Use helper functions that take tensors as parameter to avoid borrow conflicts
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

/// Actor-Critic network for PPO
#[allow(dead_code)]
pub struct ActorCriticNetwork {
    // Shared feature encoder
    fc1: Linear,
    ln1: LayerNorm,
    fc2: Linear,
    ln2: LayerNorm,
    fc3: Linear,
    ln3: LayerNorm,
    // Actor head (policy)
    actor_discrete: Linear,
    actor_param_mean: Linear,
    actor_param_logstd: Var,  // Learnable log std

    // Critic head (value function)
    critic_fc1: Linear,
    critic_fc2: Linear,

    device: Device,
    num_actions: usize,
    num_params: usize,
}


impl ActorCriticNetwork {

    pub fn new(
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        vb: VarBuilder,
    ) -> candle_core::error::Result<Self> {
        let device = vb.device().clone();
        // Shared encoder
        let fc1 = linear(state_dim, 512, vb.pp("fc1"))?;
        let ln1 = layer_norm(512, 1e-5, vb.pp("ln1"))?;
        let fc2 = linear(512, 256, vb.pp("fc2"))?;
        let ln2 = layer_norm(256, 1e-5, vb.pp("ln2"))?;
        let fc3 = linear(256, 128, vb.pp("fc3"))?;
        let ln3 = layer_norm(128, 1e-5, vb.pp("ln3"))?;

        // Actor
        let actor_discrete = linear(128, num_actions, vb.pp("actor_discrete"))?;
        let actor_param_mean = linear(128, num_params, vb.pp("actor_param_mean"))?;

        // Initialize learnable log std
        let logstd_init = Tensor::from_vec(
            vec![-1.0f32; num_params],
            &[num_params],
            &device
        )?;
        let actor_param_logstd = Var::from_tensor(&logstd_init)?;

        // Critic
        let critic_fc1 = linear(128, 64, vb.pp("critic_fc1"))?;
        let critic_fc2 = linear(64, 1, vb.pp("critic_fc2"))?;

        Ok(Self {
            fc1, ln1, fc2, ln2, fc3, ln3,
            actor_discrete,
            actor_param_mean,
            actor_param_logstd,
            critic_fc1,
            critic_fc2,
            device,
            num_actions,
            num_params,
        })
    }

    pub fn forward(
        &self,
        state: &Tensor,
        _training: bool,
    ) -> candle_core::error::Result<(Tensor, Tensor, Tensor, Tensor)> {
        // Shared features
        let mut x = self.fc1.forward(state)?;
        x = self.ln1.forward(&x)?;
        x = x.relu()?;

        x = self.fc2.forward(&x)?;
        x = self.ln2.forward(&x)?;
        x = x.relu()?;

        x = self.fc3.forward(&x)?;
        x = self.ln3.forward(&x)?;
        let features = x.relu()?;

        // Actor outputs
        let action_logits = self.actor_discrete.forward(&features)?;
        let param_mean = self.actor_param_mean.forward(&features)?.tanh()?;
        let param_std = self.actor_param_logstd.as_tensor().exp()?;

        // Critic output
        let mut value = self.critic_fc1.forward(&features)?;
        value = value.relu()?;
        let value = self.critic_fc2.forward(&value)?.squeeze(1)?;

        Ok((action_logits, param_mean, param_std, value))
    }

    /// Save PPO model to file with metadata
    pub fn save_to_file(&self, path: &Path, metadata: ModelMetadata) -> Result<()> {
        // This method already exists in the file, just ensure it's properly visible
        // The implementation around line 130-220 is already correct
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

        // Collect all tensors
        let mut tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();

        // Save all network components using helper functions
        save_linear_helper(&mut tensors, "fc1", &self.fc1)?;
        save_layernorm_helper(&mut tensors, "ln1", &self.ln1)?;
        save_linear_helper(&mut tensors, "fc2", &self.fc2)?;
        save_layernorm_helper(&mut tensors, "ln2", &self.ln2)?;
        save_linear_helper(&mut tensors, "fc3", &self.fc3)?;
        save_layernorm_helper(&mut tensors, "ln3", &self.ln3)?;

        save_linear_helper(&mut tensors, "actor_discrete", &self.actor_discrete)?;
        save_linear_helper(&mut tensors, "actor_param_mean", &self.actor_param_mean)?;

        // Save learnable log std
        let logstd_tensor = self.actor_param_logstd.as_tensor();
        let logstd_shape = logstd_tensor.dims().to_vec();
        let logstd_data = logstd_tensor.flatten_all()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        tensors.insert("actor_param_logstd".to_string(), (logstd_shape, logstd_data));

        save_linear_helper(&mut tensors, "critic_fc1", &self.critic_fc1)?;
        save_linear_helper(&mut tensors, "critic_fc2", &self.critic_fc2)?;

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
        tracing::info!("PPO model saved: {} bytes", file_size);

        Ok(())
    }

    /// Load PPO model from file - returns network and varmap
    pub fn load_from_file(
        path: &Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> Result<(Self, VarMap)> {  // FIXED: Return tuple
        use std::fs::File;
        use std::io::Read;

        tracing::info!("Loading PPO model from: {}", path.display());

        let mut file = File::open(path)?;

        // Read metadata
        let mut metadata_len_bytes = [0u8; 8];
        file.read_exact(&mut metadata_len_bytes)?;
        let metadata_len = u64::from_le_bytes(metadata_len_bytes) as usize;
        if metadata_len > 10 * 1024 * 1024 {
            return Err(crate::ExtractionError::ParseError(format!("Invalid model file: metadata length {} is too large", metadata_len)));
        }

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

        // Create network first to populate varmap with correct keys, then overwrite with loaded values
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let mut network = ActorCriticNetwork::new(state_dim, num_actions, num_params, vb)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        for (name, tensor) in tensors.iter() {
            if name == "actor_param_logstd" {
                network.actor_param_logstd = Var::from_tensor(tensor)
                    .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
            } else {
                varmap.set_one(name, tensor)
                    .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
            }
        }

        Ok((network, varmap))
    }

    /// Update load_with_device to use load_from_file
    pub fn load_with_device(
        path: &Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> Result<(Self, VarMap)> {
        Self::load_from_file(path, state_dim, num_actions, num_params, device)
    }

    /// Save to SafeTensors format
    #[allow(dead_code)]
    pub(crate) fn save_to_safetensors(&self, path: &PathBuf) -> Result<()> {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let mut tensors_data: HashMap<String, TensorView> = HashMap::new();
        let mut all_tensor_bytes: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

        // Collect all tensors
        let mut collect_tensor = |name: &str, tensor: &Tensor| -> Result<()> {
            let shape = tensor.dims().to_vec();
            let data = tensor.flatten_all()
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?
                .to_vec1::<f32>()
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
            let bytes: Vec<u8> = data.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();

            all_tensor_bytes.push((name.to_string(), shape, bytes));
            Ok(())
        };

        // Save all network components
        collect_tensor("fc1.weight", self.fc1.weight())?;
        if let Some(bias) = self.fc1.bias() {
            collect_tensor("fc1.bias", bias)?;
        }

        collect_tensor("ln1.weight", self.ln1.weight())?;
        if let Some(bias) = self.ln1.bias() {
            collect_tensor("ln1.bias", bias)?;
        }

        collect_tensor("fc2.weight", self.fc2.weight())?;
        if let Some(bias) = self.fc2.bias() {
            collect_tensor("fc2.bias", bias)?;
        }

        collect_tensor("ln2.weight", self.ln2.weight())?;
        if let Some(bias) = self.ln2.bias() {
            collect_tensor("ln2.bias", bias)?;
        }

        collect_tensor("fc3.weight", self.fc3.weight())?;
        if let Some(bias) = self.fc3.bias() {
            collect_tensor("fc3.bias", bias)?;
        }

        collect_tensor("ln3.weight", self.ln3.weight())?;
        if let Some(bias) = self.ln3.bias() {
            collect_tensor("ln3.bias", bias)?;
        }

        collect_tensor("actor_discrete.weight", self.actor_discrete.weight())?;
        if let Some(bias) = self.actor_discrete.bias() {
            collect_tensor("actor_discrete.bias", bias)?;
        }

        collect_tensor("actor_param_mean.weight", self.actor_param_mean.weight())?;
        if let Some(bias) = self.actor_param_mean.bias() {
            collect_tensor("actor_param_mean.bias", bias)?;
        }

        collect_tensor("actor_param_logstd", self.actor_param_logstd.as_tensor())?;

        collect_tensor("critic_fc1.weight", self.critic_fc1.weight())?;
        if let Some(bias) = self.critic_fc1.bias() {
            collect_tensor("critic_fc1.bias", bias)?;
        }

        collect_tensor("critic_fc2.weight", self.critic_fc2.weight())?;
        if let Some(bias) = self.critic_fc2.bias() {
            collect_tensor("critic_fc2.bias", bias)?;
        }

        // Convert to SafeTensors format
        for (name, shape, bytes) in &all_tensor_bytes {
            tensors_data.insert(
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?
            );
        }

        let serialized = safetensors::serialize(&tensors_data, None)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        std::fs::write(path, serialized)?;

        tracing::info!("PPO model saved to SafeTensors: {} bytes",
                   std::fs::metadata(path).map(|m| m.len()).unwrap_or(0));

        Ok(())
    }

    /// Save to ONNX format with metadata
    #[allow(dead_code)]
    pub(crate) fn save_to_onnx_with_metadata(&self, path: &Path, metadata: ModelMetadata) -> Result<()> {
        self.save_to_file(path, metadata)
    }

}

/// PPO Agent
pub struct PPOAgent {
    network: ActorCriticNetwork,
    optimizer: AdamW,
    #[allow(dead_code)]
    varmap: VarMap,
    // PPO hyperparameters
    clip_epsilon: f32,
    gae_lambda: f32,
    value_loss_coef: f32,
    entropy_coef: f32,
    ppo_epochs: usize,

    num_actions: usize,
    num_params: usize,
    gamma: f32,
    step_count: usize,
    device: Device,
}

impl PPOAgent {
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        gamma: f32,
        lr: f64,
        device: &Device,
        varmap: VarMap,
    ) -> Result<Self> {
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let network = ActorCriticNetwork::new(state_dim, num_actions, num_params, vb)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        let trainable_vars = varmap.all_vars();
        let params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };

        let optimizer = AdamW::new(trainable_vars, params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(Self {
            network,
            optimizer,
            varmap,
            clip_epsilon: 0.2,
            gae_lambda: 0.95,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            ppo_epochs: 4,
            num_actions,
            num_params,
            gamma,
            step_count: 0,
            device: device.clone(),
        })
    }

    /// Calculate Generalized Advantage Estimation (GAE)
    fn calculate_gae(
        &self,
        rewards: &[f32],
        values: &[f32],
        next_value: f32,
        dones: &[bool],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut advantages = vec![0.0; rewards.len()];
        let mut returns = vec![0.0; rewards.len()];

        let mut gae = 0.0;
        let mut next_val = next_value;

        for t in (0..rewards.len()).rev() {
            let done_mask = if dones[t] { 0.0 } else { 1.0 };
            let delta = rewards[t] + self.gamma * next_val * done_mask - values[t];
            gae = delta + self.gamma * self.gae_lambda * done_mask * gae;
            advantages[t] = gae;
            returns[t] = gae + values[t];
            next_val = values[t];
        }

        (advantages, returns)
    }

    /// Calculate log probability for discrete action
    fn discrete_log_prob(
        logits: &Tensor,
        actions: &Tensor,
    ) -> candle_core::error::Result<Tensor> {
        let log_probs = candle_nn::ops::log_softmax(logits, 1)?;
        log_probs.gather(&actions.unsqueeze(1)?, 1)?.squeeze(1)
    }

    /// Calculate log probability for continuous actions (Gaussian)
    fn continuous_log_prob(
        mean: &Tensor,
        std: &Tensor,
        actions: &Tensor,
    ) -> candle_core::error::Result<Tensor> {
        // Get dimensions
        let batch_size = mean.dims()[0];
        let num_params = mean.dims()[1];

        // Broadcast std to match mean shape
        // std is [num_params], need [batch_size, num_params]
        let std_broadcast = std.unsqueeze(0)?.broadcast_as(mean.shape())?;
        let variance = std_broadcast.sqr()?;
        let diff = (actions - mean)?;

        // Create pi constant with proper shape [batch_size, num_params]
        let pi_constant = Tensor::new(
            vec![2.0 * std::f32::consts::PI; batch_size * num_params],
            mean.device()
        )?.reshape(&[batch_size, num_params])?;

        let log_prob = -0.5 * (
            diff.sqr()?.div(&variance)? +
                variance.log()? +
                pi_constant.log()?
        )?;

        log_prob?.sum(1)
    }

    /// Calculate entropy for exploration bonus
    fn calculate_entropy(
        logits: &Tensor,
        std: &Tensor,
    ) -> candle_core::error::Result<Tensor> {
        // Discrete entropy
        let probs = candle_nn::ops::softmax(logits, 1)?;
        let log_probs = candle_nn::ops::log_softmax(logits, 1)?;
        let discrete_entropy = -1.0 * (probs * log_probs)?.sum(1)?.mean_all()?;

        // Continuous entropy (Gaussian)
        // std is [num_params], create constant with same shape
        let num_params = std.dims()[0];
        let constant = Tensor::new(
            vec![0.5 * (1.0 + 2.0 * std::f32::consts::PI).ln(); num_params],
            std.device()
        )?;

        let continuous_entropy = (std.log()? + constant)?.mean_all()?;

        discrete_entropy + continuous_entropy
    }

    /// PPO update step
    fn ppo_update(
        &mut self,
        states: &Tensor,
        actions_discrete: &Tensor,
        actions_continuous: &Tensor,
        old_log_probs: &Tensor,
        advantages: &Tensor,
        returns: &Tensor,
    ) -> Result<(f32, f32, f32)> {
        // Forward pass
        let (action_logits, param_mean, param_std, values) =
            self.network.forward(states, true)
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Calculate current log probabilities
        let log_probs_discrete = Self::discrete_log_prob(&action_logits, actions_discrete)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        let log_probs_continuous = Self::continuous_log_prob(&param_mean, &param_std, actions_continuous)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
        let log_probs = (log_probs_discrete + log_probs_continuous)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // PPO clipped objective
        let ratio = (log_probs.clone() - old_log_probs)?.exp()?;

        // FIXED: Normalize advantages with proper shape handling
        let batch_size = advantages.dims()[0];

        // Calculate mean and std as scalars
        let adv_mean_scalar = advantages.mean_all()?.to_scalar::<f32>()?;
        let adv_variance = advantages.sub(&Tensor::new(&[adv_mean_scalar], advantages.device())?.broadcast_as(advantages.shape())?)?.sqr()?.mean_all()?;
        let adv_std_scalar = (adv_variance.to_scalar::<f32>()? + 1e-8).sqrt();

        // Create broadcast-able tensors
        let adv_mean_broadcast = Tensor::new(vec![adv_mean_scalar; batch_size], advantages.device())?;
        let adv_std_broadcast = Tensor::new(vec![adv_std_scalar; batch_size], advantages.device())?;

        // Now shapes match: [batch_size] - [batch_size] / [batch_size]
        let advantages_norm = ((advantages - &adv_mean_broadcast)? / &adv_std_broadcast)?;

        let surr1 = (ratio.clone() * &advantages_norm)?;

        let ratio_clipped = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)?;
        let surr2 = (ratio_clipped * advantages_norm)?;

        let policy_loss = (-1.0 * surr1.minimum(&surr2)?.mean_all()?)?;

        // Value loss with clipping
        let value_loss = (values - returns)?.sqr()?.mean_all()?;

        // Entropy bonus
        let entropy = Self::calculate_entropy(&action_logits, &param_std)?;

        // Total loss - combine as scalars
        let value_loss_weighted = value_loss.to_scalar::<f32>()? * self.value_loss_coef;
        let entropy_weighted = entropy.to_scalar::<f32>()? * self.entropy_coef;
        let policy_loss_scalar = policy_loss.to_scalar::<f32>()?;

        let total_loss_scalar = policy_loss_scalar + value_loss_weighted - entropy_weighted;

        // Create tensor from combined scalar for backward pass
        let total_loss = Tensor::new(&[total_loss_scalar], policy_loss.device())?;

        // Backward and optimize
        let grads = total_loss.backward()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        self.optimizer.step(&grads)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok((
            policy_loss_scalar,
            value_loss.to_scalar::<f32>()?,
            entropy.to_scalar::<f32>()?,
        ))
    }

    pub fn load_with_device(
        path: &Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> Result<Self> {
        let (network, varmap) = ActorCriticNetwork::load_from_file(
            path, state_dim, num_actions, num_params, device
        )?;

        // Create optimizers
        let trainable_vars = varmap.all_vars();
        let params = ParamsAdamW {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };

        let optimizer = AdamW::new(trainable_vars, params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(Self {
            network,
            optimizer,
            varmap,
            clip_epsilon: 0.2,
            gae_lambda: 0.95,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            ppo_epochs: 4,
            num_actions,
            num_params,
            gamma: 0.95,
            step_count: 0,
            device: device.clone(),
        })
    }
}
impl RLAgent for PPOAgent {
    fn select_action(&self, state: &[f32], _epsilon: f32) -> Result<(usize, Vec<f32>)> {
        // PPO uses stochastic policy, not epsilon-greedy
        let state_tensor = Tensor::from_vec(
            state.to_vec(),
            &[1, state.len()],
            &self.device
        )?;
        let (action_logits, param_mean, param_std, _value) =
            self.network.forward(&state_tensor, false)?;

        // Sample discrete action from categorical distribution
        let probs = candle_nn::ops::softmax(&action_logits, 1)?.to_vec2::<f32>()?;
        let discrete_action = sample_categorical(&probs[0]);

        // Sample continuous parameters from Gaussian
        let mean_vec = param_mean.to_vec2::<f32>()?;
        let std_vec = param_std.to_vec1::<f32>()?;
        let continuous_params = sample_gaussian(&mean_vec[0], &std_vec);

        Ok((discrete_action, continuous_params))
    }

    fn save_with_metadata(
        &self,
        path: &Path,
        training_episodes: usize,
        hyperparameters: HashMap<String, f64>,
    ) -> Result<()> {
        let metadata = ModelMetadata::new(
            300,
            self.num_actions,
            self.num_params,
            AlgorithmType::PPO,
            training_episodes,
            hyperparameters,
        );

        self.network.save_to_file(path, metadata)
    }

    fn save(&self, path: &Path) -> Result<()> {
        self.save_with_metadata(path, 0, std::collections::HashMap::new())
    }

    fn train_step(
        &mut self,
        replay_buffer: &mut PrioritizedReplayBuffer,
        batch_size: usize,
    ) -> Result<f32> {
        let batch = replay_buffer.sample(batch_size);
        if batch.is_none() {
            return Ok(0.0);
        }

        let batch = batch.unwrap();
        let experiences = &batch.experiences;

        if experiences.is_empty() {
            return Ok(0.0);
        }

        // Convert experiences to tensors
        let state_dim = experiences[0].state.len();
        let states_flat: Vec<f32> = experiences.iter()
            .flat_map(|e| e.state.clone())
            .collect();
        let states_tensor = Tensor::from_vec(
            states_flat,
            &[experiences.len(), state_dim],
            &self.device
        )?;

        // Get old policy values
        let (old_logits, old_means, old_stds, old_values) =
            self.network.forward(&states_tensor, false)?;

        // Extract actions
        let actions_discrete: Vec<i64> = experiences.iter()
            .map(|e| e.action.0 as i64)
            .collect();
        let actions_discrete_tensor = Tensor::from_vec(
            actions_discrete,
            &[experiences.len()],
            &self.device
        )?;

        let actions_continuous_flat: Vec<f32> = experiences.iter()
            .flat_map(|e| e.action.1.clone())
            .collect();
        let actions_continuous_tensor = Tensor::from_vec(
            actions_continuous_flat,
            &[experiences.len(), self.num_params],
            &self.device
        )?;

        // Calculate old log probabilities
        let old_log_probs_discrete = Self::discrete_log_prob(&old_logits, &actions_discrete_tensor)?;
        let old_log_probs_continuous = Self::continuous_log_prob(&old_means, &old_stds, &actions_continuous_tensor)?;
        let old_log_probs = (old_log_probs_discrete + old_log_probs_continuous)?;

        // Calculate GAE
        let rewards: Vec<f32> = experiences.iter().map(|e| e.reward).collect();
        let values_vec: Vec<f32> = old_values.to_vec1()?;
        let dones: Vec<bool> = experiences.iter().map(|e| e.done).collect();

        let (advantages, returns) = self.calculate_gae(
            &rewards,
            &values_vec,
            0.0,
            &dones,
        );

        let advantages_tensor = Tensor::from_vec(advantages, &[experiences.len()], &self.device)?;
        let returns_tensor = Tensor::from_vec(returns, &[experiences.len()], &self.device)?;

        // PPO update for multiple epochs
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut _total_entropy = 0.0;

        for _ in 0..self.ppo_epochs {
            let (policy_loss, value_loss, entropy) = self.ppo_update(
                &states_tensor,
                &actions_discrete_tensor,
                &actions_continuous_tensor,
                &old_log_probs,
                &advantages_tensor,
                &returns_tensor,
            )?;

            total_policy_loss += policy_loss;
            total_value_loss += value_loss;
            _total_entropy += entropy;
        }

        self.step_count += 1;

        let avg_loss = (total_policy_loss + total_value_loss) / self.ppo_epochs as f32;
        Ok(avg_loss)
    }

    fn update_target_network(&mut self) {
        // PPO doesn't use target networks
    }

    fn get_step_count(&self) -> usize {
        self.step_count
    }

    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::PPO
    }

    fn get_info(&self) -> AgentInfo {
        AgentInfo {
            algorithm: AlgorithmType::PPO,
            num_parameters: 0, // TODO: calculate
            state_dim: 0,
            num_actions: self.num_actions,
            continuous_params: self.num_params,
            version: "1.0.0".to_string(),
            features: vec![
                "actor_critic".to_string(),
                "clipped_objective".to_string(),
                "gae".to_string(),
                "entropy_bonus".to_string(),
            ],
        }
    }
}

// ADDITIONAL HELPER: Debug tensor shapes (for development)
// Usage in ppo_update for debugging:
// debug_tensor_shape("advantages", advantages);
// debug_tensor_shape("adv_mean", &adv_mean);
// debug_tensor_shape("adv_std", &adv_std);

#[cfg(debug_assertions)]
#[allow(dead_code)]
fn debug_tensor_shape(name: &str, tensor: &Tensor) {
    eprintln!("DEBUG: {} shape: {:?}", name, tensor.dims());
}

#[cfg(not(debug_assertions))]
fn debug_tensor_shape(_name: &str, _tensor: &Tensor) {
    // No-op in release builds
}
