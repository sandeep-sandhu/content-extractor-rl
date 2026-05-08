// ============================================================================
// FILE: crates/content-extractor-rl/src/agents/dqn_agent.rs
// ============================================================================

use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, Optimizer, AdamW, ParamsAdamW, VarMap};
use crate::models::{DuelingDQN, NetworkConfig};
use crate::replay_buffer::{PrioritizedReplayBuffer, SampledBatch};
use crate::{Result, agents::{RLAgent, AlgorithmType, AgentInfo}};
use rand::RngExt;
use tracing::{info, warn};
use std::path::Path;

/// DQN Agent for article extraction:
pub struct DQNAgent {
    pub(crate) online_network: DuelingDQN,
    target_network: DuelingDQN,
    optimizer: AdamW,
    varmap: VarMap,
    num_actions: usize,
    num_params: usize,
    gamma: f32,
    step_count: usize,
    device: Device,
}

impl DQNAgent {
    /// Create new DQN agent with custom network configuration
    pub fn new(
        network_config: NetworkConfig,
        gamma: f32,
        lr: f64,
        device: &Device,
        varmap: VarMap,
    ) -> Result<Self> {
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let online_network = DuelingDQN::new(
            network_config.state_dim,
            network_config.num_actions,
            network_config.num_params,
            vb.pp("online")
        )?;

        let target_varmap = VarMap::new();
        let target_vb = VarBuilder::from_varmap(&target_varmap, DType::F32, device);
        let mut target_network = DuelingDQN::new(
            network_config.state_dim,
            network_config.num_actions,
            network_config.num_params,
            target_vb.pp("target")
        )?;

        // Get trainable variables from the varmap
        let trainable_vars = varmap.all_vars();

        let params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
        };

        let optimizer = AdamW::new(trainable_vars, params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Copy online network weights to target network initially
        target_network.copy_weights_from(&online_network)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(Self {
            online_network,
            target_network,
            optimizer,
            varmap,
            num_actions: network_config.num_actions,
            num_params: network_config.num_params,
            gamma,
            step_count: 0,
            device: device.clone(),
        })
    }

    /// Copy weights from source network to target network
    fn copy_network_weights(source: &DuelingDQN, target: &mut DuelingDQN) -> Result<()> {
        target.copy_weights_from(source)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))
    }

    /// Update target network using soft update
    pub fn update_target_network(&mut self) {
        // Implement hard update (full copy of weights)
        // For soft update with tau, you would blend: target = tau * online + (1-tau) * target
        if let Err(e) = Self::copy_network_weights(&self.online_network, &mut self.target_network) {
            warn!("Failed to update target network: {}", e);
        } else {
            info!("Target network updated (hard update)");
        }
    }

    /// Get step count
    pub fn get_step_count(&self) -> usize {
        self.step_count
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: &[f32], epsilon: f32) -> Result<(usize, Vec<f32>)> {
        let mut rng = rand::rng();

        if rng.random::<f32>() < epsilon {
            let discrete_action = rng.random_range(0..self.num_actions);
            let params: Vec<f32> = (0..self.num_params)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            Ok((discrete_action, params))
        } else {
            // Greedy action
            let state_tensor = Tensor::from_vec(state.to_vec(), &[1, state.len()], &self.device)
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

            let (q_values, param_mean, _param_std) = self.online_network.forward(&state_tensor, false)
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

            // Get discrete action
            let q_vals = q_values.to_vec2::<f32>()
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
            let discrete_action = q_vals[0].iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Get continuous params
            let params = param_mean.to_vec2::<f32>()
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
            let continuous_params = params[0].clone();

            Ok((discrete_action, continuous_params))
        }
    }

    /// Complete training step with proper loss calculation
    pub fn train_step(&mut self, replay_buffer: &mut PrioritizedReplayBuffer, batch_size: usize) -> Result<f32> {
        let batch = replay_buffer.sample(batch_size);

        if batch.is_none() {
            return Ok(0.0);
        }

        let SampledBatch { experiences, indices, weights } = batch.unwrap();

        // Extract components from experiences
        let states: Vec<Vec<f32>> = experiences.iter()
            .map(|e| e.state.clone())
            .collect();
        let actions_discrete: Vec<usize> = experiences.iter()
            .map(|e| e.action.0)
            .collect();
        let actions_params: Vec<Vec<f32>> = experiences.iter()
            .map(|e| e.action.1.clone())
            .collect();
        let rewards: Vec<f32> = experiences.iter()
            .map(|e| e.reward)
            .collect();
        let next_states: Vec<Vec<f32>> = experiences.iter()
            .map(|e| e.next_state.clone())
            .collect();
        let dones: Vec<f32> = experiences.iter()
            .map(|e| if e.done { 1.0 } else { 0.0 })
            .collect();

        // Convert to tensors
        let state_dim = states[0].len();
        let states_flat: Vec<f32> = states.into_iter().flatten().collect();
        let states_tensor = Tensor::from_vec(
            states_flat,
            &[batch_size, state_dim],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let next_states_flat: Vec<f32> = next_states.into_iter().flatten().collect();
        let next_states_tensor = Tensor::from_vec(
            next_states_flat,
            &[batch_size, state_dim],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let rewards_tensor = Tensor::from_vec(
            rewards,
            &[batch_size],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let dones_tensor = Tensor::from_vec(
            dones,
            &[batch_size],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let weights_tensor = Tensor::from_vec(
            weights,
            &[batch_size],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Actions tensors
        let actions_discrete_tensor = Tensor::from_vec(
            actions_discrete.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            &[batch_size],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let actions_params_flat: Vec<f32> = actions_params.into_iter().flatten().collect();
        let actions_params_tensor = Tensor::from_vec(
            actions_params_flat,
            &[batch_size, self.num_params],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Forward pass through online network
        let (q_values, param_means, param_stds) = self.online_network.forward(&states_tensor, true)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // VALIDATION: Check for NaN/Inf in forward pass
        let q_sample = q_values.get(0)?.to_vec1::<f32>()?;
        if q_sample.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(crate::ExtractionError::ModelError(
                "NaN/Inf detected in Q-values forward pass".to_string()
            ));
        }

        // Gather Q-values for taken actions
        let q_values_selected = q_values
            .gather(&actions_discrete_tensor.unsqueeze(1)?, 1)?
            .squeeze(1)?;

        // Double DQN: Use online network to select actions, target network to evaluate
        let (next_q_online, _, _) = self.online_network.forward(&next_states_tensor, false)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let next_actions = next_q_online.argmax(1)?;

        let (next_q_target, _, _) = self.target_network.forward(&next_states_tensor, false)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let next_q_values = next_q_target
            .gather(&next_actions.unsqueeze(1)?, 1)?
            .squeeze(1)?;

        // Calculate TD targets with proper shape broadcasting
        let ones = Tensor::ones(&[batch_size], DType::F32, &self.device)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Create gamma tensor with same shape as batch
        let gamma_vec = vec![self.gamma; batch_size];
        let gamma_tensor = Tensor::from_vec(gamma_vec, &[batch_size], &self.device)?;

        // Calculate discount factors: gamma * (1 - done)
        let discount_factors = (ones - dones_tensor)?
            .mul(&gamma_tensor)?;

        // TD target: reward + gamma * (1 - done) * next_q
        let td_targets = rewards_tensor
            .add(&next_q_values.mul(&discount_factors)?)?;

        // Calculate TD errors for priority update
        let td_errors_tensor = (td_targets.clone() - q_values_selected.clone())?;
        let td_errors: Vec<f32> = td_errors_tensor
            .to_vec1()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Q-value loss (Smooth L1 / Huber loss)
        let q_loss_elements = smooth_l1_loss(&q_values_selected, &td_targets)?;
        let weighted_q_loss = (q_loss_elements * weights_tensor.clone())?;
        let loss_q = weighted_q_loss.mean_all()?;

        // Parameter loss (Negative log-likelihood of Gaussian)
        let param_loss = self.calculate_param_loss(&param_means, &param_stds, &actions_params_tensor)?;

        // Combine losses
        let loss_q_scalar = loss_q.to_scalar::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let param_loss_scalar = param_loss.to_scalar::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let total_loss_scalar = loss_q_scalar + 0.1 * param_loss_scalar;

        // Create tensor from combined scalar
        let total_loss = Tensor::from_vec(
            vec![total_loss_scalar],
            &[1],
            &self.device
        )?;

        // VALIDATION: Check final loss
        if total_loss_scalar.is_nan() || total_loss_scalar.is_infinite() {
            return Err(crate::ExtractionError::ModelError(
                format!("Invalid loss: {}", total_loss_scalar)
            ));
        }

        // Get GradStore from backward pass
        // Perform backward pass to get gradients
        let mut grad_store = total_loss.backward()?;

        // IMPROVED: Clip gradients to prevent explosion
        // Get all trainable variables from the varmap
        let vars = self.varmap.all_vars();
        let max_grad_norm = 1.0f32;
        let mut total_norm_sq = 0.0f32;

        // Calculate total gradient norm for all trainable variables
        for var in &vars {
            if let Some(grad) = grad_store.get(var) {
                let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
                total_norm_sq += norm_sq;
            }
        }

        let total_norm = total_norm_sq.sqrt();

        // Apply gradient clipping if needed
        if total_norm > max_grad_norm {
            let clip_coef = max_grad_norm / (total_norm + 1e-6);

            // Apply clipping to each gradient
            for var in self.varmap.all_vars() {
                if let Some(grad) = grad_store.get(&var) {
                    // Create a tensor for the clip coefficient
                    let clip_coef_tensor = Tensor::from_vec(
                        vec![clip_coef],
                        &[1],
                        &self.device
                    )?;

                    // Multiply gradient by clip coefficient
                    let clipped_grad = grad.mul(&clip_coef_tensor)?;

                    // Update the gradient in the grad store
                    grad_store.insert(&var, clipped_grad);
                }
            }

            if self.step_count.is_multiple_of(1000) {
                info!("Gradient norm: {:.4}, clipped with coef: {:.4}", total_norm, clip_coef);
            }
        }

        // Step the optimizer with gradients
        self.optimizer.step(&grad_store)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Update priorities in replay buffer
        replay_buffer.update_priorities(&indices, &td_errors);

        self.step_count += 1;

        // Return loss value
        Ok(total_loss_scalar)
    }

    /// Calculate parameter loss (negative log-likelihood)
    fn calculate_param_loss(
        &self,
        means: &Tensor,
        stds: &Tensor,
        actions: &Tensor,
    ) -> candle_core::error::Result<Tensor> {
        let batch_size = actions.dims()[0];
        let num_params = actions.dims()[1];

        let diff = actions.sub(means)?;

        // Broadcast stds to match batch dimension
        let stds_broadcast = stds.unsqueeze(0)?.broadcast_as(means.shape())?;

        let variance = stds_broadcast.sqr()?;
        let squared_diff = diff.sqr()?.div(&variance)?;

        let log_std = stds_broadcast.log()?;

        // Create constant tensors with proper shapes
        let pi_vec = vec![std::f32::consts::PI; batch_size * num_params];
        let pi_constant = Tensor::from_vec(pi_vec, &[batch_size, num_params], &self.device)?;

        let half_vec = vec![0.5f32; batch_size * num_params];
        let half_tensor = Tensor::from_vec(half_vec, &[batch_size, num_params], &self.device)?;

        let constant = pi_constant.log()?.mul(&half_tensor)?;

        let nll = constant
            .add(&log_std)?
            .add(&squared_diff.mul(&half_tensor)?)?;

        nll.mean_all()
    }

    /// Save model in both ONNX and SafeTensors formats
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        self.online_network.save_to_onnx(path)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let safetensors_path = path.with_extension("safetensors");
        self.online_network.save_to_safetensors(&safetensors_path)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        tracing::info!("Model saved: ONNX ({} bytes), SafeTensors ({} bytes)",
                   std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
                   std::fs::metadata(&safetensors_path).map(|m| m.len()).unwrap_or(0));

        Ok(())
    }

    /// Load model from ONNX format
    pub fn load(
        path: &std::path::Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
    ) -> Result<Self> {
        let device = crate::device::get_device();
        Self::load_with_device(path, state_dim, num_actions, num_params, &device)
    }

    pub fn load_with_device(
        path: &std::path::Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> Result<Self> {
        tracing::info!("Loading model on device: {}", crate::device::get_device_info(device));

        let online_network = DuelingDQN::load_from_onnx(path, state_dim, num_actions, num_params, device)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Create target network on the SAME device
        let target_varmap = VarMap::new();
        let vb_target = VarBuilder::from_varmap(&target_varmap, DType::F32, device);
        let target_network = DuelingDQN::new(state_dim, num_actions, num_params, vb_target)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let varmap = VarMap::new();
        let vars = varmap.all_vars();
        let params = ParamsAdamW::default();
        let optimizer = AdamW::new(vars, params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(Self {
            online_network,
            target_network,
            optimizer,
            varmap,
            num_actions,
            num_params,
            gamma: 0.95,
            step_count: 0,
            device: device.clone(),
        })
    }
}

// Implement RLAgent trait for DQNAgent
impl RLAgent for DQNAgent {
    fn select_action(&self, state: &[f32], epsilon: f32) -> Result<(usize, Vec<f32>)> {
        let mut rng = rand::rng();

        if rng.random::<f32>() < epsilon {
            let discrete_action = rng.random_range(0..self.num_actions);
            let params: Vec<f32> = (0..self.num_params)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            Ok((discrete_action, params))
        } else {
            let state_tensor = Tensor::from_vec(state.to_vec(), &[1, state.len()], &self.device)
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

            let (q_values, param_mean, _param_std) = self.online_network.forward(&state_tensor, false)
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

            let q_vals = q_values.to_vec2::<f32>()
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
            let discrete_action = q_vals[0].iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let params = param_mean.to_vec2::<f32>()
                .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;
            let continuous_params = params[0].clone();

            Ok((discrete_action, continuous_params))
        }
    }


    /// Complete training step with proper loss calculation
    fn train_step(&mut self, replay_buffer: &mut PrioritizedReplayBuffer, batch_size: usize) -> Result<f32> {
        let batch = replay_buffer.sample(batch_size);

        if batch.is_none() {
            return Ok(0.0);
        }

        let SampledBatch { experiences, indices, weights } = batch.unwrap();

        // Extract components from experiences
        let states: Vec<Vec<f32>> = experiences.iter()
            .map(|e| e.state.clone())
            .collect();
        let actions_discrete: Vec<usize> = experiences.iter()
            .map(|e| e.action.0)
            .collect();
        let actions_params: Vec<Vec<f32>> = experiences.iter()
            .map(|e| e.action.1.clone())
            .collect();
        let rewards: Vec<f32> = experiences.iter()
            .map(|e| e.reward)
            .collect();
        let next_states: Vec<Vec<f32>> = experiences.iter()
            .map(|e| e.next_state.clone())
            .collect();
        let dones: Vec<f32> = experiences.iter()
            .map(|e| if e.done { 1.0 } else { 0.0 })
            .collect();

        // Convert to tensors
        let state_dim = states[0].len();
        let states_flat: Vec<f32> = states.into_iter().flatten().collect();
        let states_tensor = Tensor::from_vec(
            states_flat,
            &[batch_size, state_dim],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let next_states_flat: Vec<f32> = next_states.into_iter().flatten().collect();
        let next_states_tensor = Tensor::from_vec(
            next_states_flat,
            &[batch_size, state_dim],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let rewards_tensor = Tensor::from_vec(
            rewards,
            &[batch_size],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let dones_tensor = Tensor::from_vec(
            dones,
            &[batch_size],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let weights_tensor = Tensor::from_vec(
            weights,
            &[batch_size],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Actions tensors
        let actions_discrete_tensor = Tensor::from_vec(
            actions_discrete.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            &[batch_size],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let actions_params_flat: Vec<f32> = actions_params.into_iter().flatten().collect();
        let actions_params_tensor = Tensor::from_vec(
            actions_params_flat,
            &[batch_size, self.num_params],
            &self.device
        ).map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Forward pass through online network
        let (q_values, param_means, param_stds) = self.online_network.forward(&states_tensor, true)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // VALIDATION: Check for NaN/Inf in forward pass
        let q_sample = q_values.get(0)?.to_vec1::<f32>()?;
        if q_sample.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err(crate::ExtractionError::ModelError(
                "NaN/Inf detected in Q-values forward pass".to_string()
            ));
        }

        // Gather Q-values for taken actions
        let q_values_selected = q_values
            .gather(&actions_discrete_tensor.unsqueeze(1)?, 1)?
            .squeeze(1)?;

        // Double DQN: Use online network to select actions, target network to evaluate
        let (next_q_online, _, _) = self.online_network.forward(&next_states_tensor, false)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let next_actions = next_q_online.argmax(1)?;

        let (next_q_target, _, _) = self.target_network.forward(&next_states_tensor, false)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let next_q_values = next_q_target
            .gather(&next_actions.unsqueeze(1)?, 1)?
            .squeeze(1)?;

        // Calculate TD targets with proper shape broadcasting
        let ones = Tensor::ones(&[batch_size], DType::F32, &self.device)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Create gamma tensor with same shape as batch
        let gamma_vec = vec![self.gamma; batch_size];
        let gamma_tensor = Tensor::from_vec(gamma_vec, &[batch_size], &self.device)?;

        // Calculate discount factors: gamma * (1 - done)
        let discount_factors = (ones - dones_tensor)?
            .mul(&gamma_tensor)?;

        // TD target: reward + gamma * (1 - done) * next_q
        let td_targets = rewards_tensor
            .add(&next_q_values.mul(&discount_factors)?)?;

        // Calculate TD errors for priority update
        let td_errors_tensor = (td_targets.clone() - q_values_selected.clone())?;
        let td_errors: Vec<f32> = td_errors_tensor
            .to_vec1()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Q-value loss (Smooth L1 / Huber loss)
        let q_loss_elements = smooth_l1_loss(&q_values_selected, &td_targets)?;
        let weighted_q_loss = (q_loss_elements * weights_tensor.clone())?;
        let loss_q = weighted_q_loss.mean_all()?;

        // Parameter loss (Negative log-likelihood of Gaussian)
        let param_loss = self.calculate_param_loss(&param_means, &param_stds, &actions_params_tensor)?;

        // Combine losses
        let loss_q_scalar = loss_q.to_scalar::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let param_loss_scalar = param_loss.to_scalar::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let total_loss_scalar = loss_q_scalar + 0.1 * param_loss_scalar;

        // Create tensor from combined scalar
        let total_loss = Tensor::from_vec(
            vec![total_loss_scalar],
            &[1],
            &self.device
        )?;

        // VALIDATION: Check final loss
        if total_loss_scalar.is_nan() || total_loss_scalar.is_infinite() {
            return Err(crate::ExtractionError::ModelError(
                format!("Invalid loss: {}", total_loss_scalar)
            ));
        }

        // Get GradStore from backward pass
        // Perform backward pass to get gradients
        let mut grad_store = total_loss.backward()?;

        // IMPROVED: Clip gradients to prevent explosion
        // Get all trainable variables from the varmap
        let vars = self.varmap.all_vars();
        let max_grad_norm = 1.0f32;
        let mut total_norm_sq = 0.0f32;

        // Calculate total gradient norm for all trainable variables
        for var in &vars {
            if let Some(grad) = grad_store.get(var) {
                let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
                total_norm_sq += norm_sq;
            }
        }

        let total_norm = total_norm_sq.sqrt();

        // Apply gradient clipping if needed
        if total_norm > max_grad_norm {
            let clip_coef = max_grad_norm / (total_norm + 1e-6);

            // Apply clipping to each gradient
            for var in self.varmap.all_vars() {
                if let Some(grad) = grad_store.get(&var) {
                    // Create a tensor for the clip coefficient
                    let clip_coef_tensor = Tensor::from_vec(
                        vec![clip_coef],
                        &[1],
                        &self.device
                    )?;

                    // Multiply gradient by clip coefficient
                    let clipped_grad = grad.mul(&clip_coef_tensor)?;

                    // Update the gradient in the grad store
                    grad_store.insert(&var, clipped_grad);
                }
            }

            if self.step_count.is_multiple_of(1000) {
                info!("Gradient norm: {:.4}, clipped with coef: {:.4}", total_norm, clip_coef);
            }
        }

        // Step the optimizer with gradients
        self.optimizer.step(&grad_store)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Update priorities in replay buffer
        replay_buffer.update_priorities(&indices, &td_errors);

        self.step_count += 1;

        // Return loss value
        Ok(total_loss_scalar)
    }

    fn update_target_network(&mut self) {
        if let Err(e) = Self::copy_network_weights(&self.online_network, &mut self.target_network) {
            warn!("Failed to update target network: {}", e);
        } else {
            info!("Target network updated (hard update)");
        }
    }

    fn get_step_count(&self) -> usize {
        self.step_count
    }
    
    fn save_with_metadata(
        &self,
        path: &Path,
        training_episodes: usize,
        hyperparameters: std::collections::HashMap<String, f64>,
    ) -> Result<()> {
        use crate::models::ModelMetadata;

        let metadata = ModelMetadata::new(
            300,  // state_dim - should get from self
            self.num_actions,
            self.num_params,
            AlgorithmType::DuelingDQN,
            training_episodes,
            hyperparameters,
        );

        self.online_network.save_to_onnx_with_metadata(path, metadata)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let safetensors_path = path.with_extension("safetensors");
        self.online_network.save_to_safetensors(&safetensors_path)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        tracing::info!("Model saved with metadata: ONNX ({} bytes), SafeTensors ({} bytes)",
               std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
               std::fs::metadata(&safetensors_path).map(|m| m.len()).unwrap_or(0));

        Ok(())
    }

    fn save(&self, path: &Path) -> Result<()> {
        // Default save without extra metadata
        self.save_with_metadata(path, 0, std::collections::HashMap::new())
    }


    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::DuelingDQN
    }

    fn get_info(&self) -> AgentInfo {
        AgentInfo {
            algorithm: AlgorithmType::DuelingDQN,
            num_parameters: 338525,
            state_dim: 300,
            num_actions: self.num_actions,
            continuous_params: self.num_params,
            version: "1.0.0".to_string(),
            features: vec![
                "dueling".to_string(),
                "double_dqn".to_string(),
                "prioritized_replay".to_string(),
            ],
        }
    }
}

/// Smooth L1 loss (Huber loss)
fn smooth_l1_loss(predicted: &Tensor, target: &Tensor) -> candle_core::error::Result<Tensor>
{
    let diff = predicted.sub(target)?;
    let abs_diff = diff.abs()?;

    let batch_size = predicted.dims()[0];
    let threshold_vec = vec![1.0f32; batch_size];
    let threshold = Tensor::from_vec(threshold_vec, &[batch_size], predicted.device())?;

    let half_vec = vec![0.5f32; batch_size];
    let half_tensor = Tensor::from_vec(half_vec, &[batch_size], predicted.device())?;

    let small_loss = diff.sqr()?.mul(&half_tensor)?;
    let large_loss = abs_diff.sub(&half_tensor)?;

    abs_diff.lt(&threshold)?
        .where_cond(&small_loss, &large_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay_buffer::{PrioritizedReplayBuffer, Experience};
    use candle_core::Device;
    use candle_nn::VarBuilder;
    use candle_core::DType;
    use crate::Config;
    use crate::models::NetworkConfig;

    fn create_network_config(config: &Config) -> NetworkConfig {
        NetworkConfig {
            state_dim: config.state_dim,
            num_actions: config.num_discrete_actions,
            num_params: config.num_continuous_params,
            hidden_layers: vec![512, 256, 128],
            use_layer_norm: true,
            dropout: 0.1,
            value_hidden: 64,
            advantage_hidden: 64,
        }
    }

    #[test]
    fn test_train_step_no_shape_mismatch() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let config = Config::default();
        let network_config = create_network_config(&config);

        let mut agent = DQNAgent::new(
            network_config,
            0.95,
            0.001,
            &device,
            varmap,
        ).unwrap();

        let mut replay_buffer = PrioritizedReplayBuffer::new(10000, 0.6, 0.4);

        for _ in 0..1000 {
            let exp = Experience {
                state: vec![0.1; 300],
                action: (0, vec![0.0; 6]),
                reward: 1.0,
                next_state: vec![0.2; 300],
                done: false,
            };
            replay_buffer.add(exp);
        }

        let result = agent.train_step(&mut replay_buffer, 512);

        match result {
            Ok(loss) => {
                println!("Training step successful, loss: {}", loss);
                assert!(!loss.is_nan(), "Loss should not be NaN");
                assert!(!loss.is_infinite(), "Loss should not be infinite");
            }
            Err(e) => {
                panic!("Training step failed: {}", e);
            }
        }
    }
}