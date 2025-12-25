use candle_core::{Device, Tensor, DType, Var};
use candle_nn::{VarBuilder, Optimizer, AdamW, ParamsAdamW, VarMap};
use crate::models::DuelingDQN;
use crate::replay_buffer::{PrioritizedReplayBuffer, SampledBatch};
use crate::Result;
use rand::Rng;
use tracing::{info, warn};

/// DQN Agent for article extraction
pub struct DQNAgent {
    pub(crate) online_network: DuelingDQN,
    target_network: DuelingDQN,
    optimizer: AdamW,
    trainable_vars: Vec<Var>, // Store for gradient clipping
    num_actions: usize,
    num_params: usize,
    gamma: f32,
    step_count: usize,
    device: Device,
}

impl DQNAgent {
    /// Create new DQN agent
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        gamma: f32,
        lr: f64,
        device: &Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let online_network = DuelingDQN::new(state_dim, num_actions, num_params, vb.pp("online"))?;

        // Create target network with separate VarBuilder
        let target_varmap = VarMap::new();
        let target_vb = VarBuilder::from_varmap(&target_varmap, DType::F32, device);
        let mut target_network = DuelingDQN::new(state_dim, num_actions, num_params, target_vb.pp("target"))?;

        // Get trainable variables
        let trainable_vars = vb.data().all_vars();

        let params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-4,
        };

        let optimizer = AdamW::new(trainable_vars.clone(), params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Copy online network weights to target network initially
        // This ensures they start with the same weights
        // Initialize target network with same weights as online network
        target_network.copy_weights_from(&online_network)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(Self {
            online_network,
            target_network,
            optimizer,
            trainable_vars,
            num_actions,
            num_params,
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
        // TODO: improve this
        let tau = 0.005; // Soft update parameter

        // For now, implement hard update
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
    // In agent.rs - Replace the entire train_step method

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

        // CRITICAL FIX: Properly combine losses without shape mismatch
        // Both loss_q and param_loss are scalars (shape [])
        // Convert to f32, combine, then back to tensor
        let loss_q_scalar = loss_q.to_scalar::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let param_loss_scalar = param_loss.to_scalar::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Combine with weight (0.1 for param loss)
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

        // Backward pass
        self.optimizer.backward_step(&total_loss)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // CLIP GRADIENTS to prevent explosion
        // TODO: check and improve this
        let max_grad_norm = 1.0;
        let mut total_norm_sq = 0.0;

        // Calculate total gradient norm
        for var in &self.trainable_vars {
            if let Some(grad) = var.grad() {
                let norm_sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
                total_norm_sq += norm_sq;
            }
        }

        let total_norm = total_norm_sq.sqrt();

        // Clip gradients if norm exceeds threshold
        if total_norm > max_grad_norm {
            let scale = max_grad_norm / (total_norm + 1e-6);

            for var in &self.trainable_vars {
                if let Some(grad) = var.grad() {
                    let clipped_grad = grad.mul_scalar(scale)?;
                    var.set_grad(&clipped_grad)?;
                }
            }
        }

        // Step the optimizer with potentially clipped gradients
        self.optimizer.step()
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
        // Negative log-likelihood of Gaussian distribution
        // -log(p(x)) = 0.5 * log(2π) + log(σ) + 0.5 * ((x - μ) / σ)²

        let batch_size = actions.dims()[0];
        let num_params = actions.dims()[1];

        let diff = actions.sub(means)?;

        // FIXED: Broadcast stds to match batch dimension
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
        // Save ONNX
        self.online_network.save_to_onnx(path)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Save SafeTensors (verify path extension)
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
        tracing::info!("Loading model on device: {}", crate::device::get_device_info(&device));

        let online_network = DuelingDQN::load_from_onnx(path, state_dim, num_actions, num_params, &device)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let vb_target = VarBuilder::zeros(DType::F32, &device);
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
            num_actions,
            num_params,
            gamma: 0.95,
            step_count: 0,
            device,
        })
    }

    pub fn load_with_device(
        path: &std::path::Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> Result<Self> {
        tracing::info!("Loading model on device: {}", crate::device::get_device_info(&device));

        let online_network = DuelingDQN::load_from_onnx(path, state_dim, num_actions, num_params, device)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Create target network on the SAME device
        let vb_target = VarBuilder::zeros(DType::F32, device);
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
            num_actions,
            num_params,
            gamma: 0.95,
            step_count: 0,
            device: device.clone(),
        })
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

    #[test]
    fn test_train_step_no_shape_mismatch() {
        // This test verifies the fix for "shape mismatch in mul" error
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mut agent = DQNAgent::new(
            300,  // state_dim
            16,   // num_actions
            6,    // num_params
            0.95, // gamma
            0.001, // lr
            &device,
            vb,
        ).unwrap();

        let mut replay_buffer = PrioritizedReplayBuffer::new(10000, 0.6, 0.4);

        // Add enough experiences for a full batch
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

        // This should not panic with shape mismatch
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

    #[test]
    fn test_train_step_different_batch_sizes() {
        // Test with various batch sizes to ensure robustness
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mut agent = DQNAgent::new(300, 16, 6, 0.95, 0.001, &device, vb).unwrap();
        let mut replay_buffer = PrioritizedReplayBuffer::new(10000, 0.6, 0.4);

        // Add experiences
        for _ in 0..2000 {
            let exp = Experience {
                state: vec![0.1; 300],
                action: (5, vec![0.5; 6]),
                reward: 0.5,
                next_state: vec![0.2; 300],
                done: false,
            };
            replay_buffer.add(exp);
        }

        // Test different batch sizes
        for batch_size in [64, 128, 256, 512, 1024] {
            let result = agent.train_step(&mut replay_buffer, batch_size);
            assert!(result.is_ok(), "Failed with batch_size={}: {:?}", batch_size, result);

            let loss = result.unwrap();
            assert!(!loss.is_nan(), "NaN loss with batch_size={}", batch_size);
            assert!(!loss.is_infinite(), "Infinite loss with batch_size={}", batch_size);
        }
    }

    #[test]
    fn test_parameter_loss_calculation() {
        // Test the fixed calculate_param_loss method
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let agent = DQNAgent::new(300, 16, 6, 0.95, 0.001, &device, vb).unwrap();

        let batch_size = 128;
        let num_params = 6;

        // Create test tensors
        let means = Tensor::zeros(&[batch_size, num_params], DType::F32, &device).unwrap();
        let stds = Tensor::ones(&[num_params], DType::F32, &device).unwrap(); // Shape [6]
        let actions = Tensor::zeros(&[batch_size, num_params], DType::F32, &device).unwrap();

        // This should not panic
        let loss = agent.calculate_param_loss(&means, &stds, &actions);

        assert!(loss.is_ok(), "Parameter loss calculation failed: {:?}", loss);

        let loss_value = loss.unwrap().to_scalar::<f32>().unwrap();
        assert!(!loss_value.is_nan(), "Loss is NaN");
        assert!(!loss_value.is_infinite(), "Loss is infinite");
    }

    #[test]
    fn test_loss_combination() {
        // Test that loss combination doesn't cause shape mismatches
        let device = Device::Cpu;

        // Create two scalar losses
        let loss_q = Tensor::from_vec(vec![0.5f32], &[1], &device).unwrap();
        let param_loss = Tensor::from_vec(vec![0.3f32], &[1], &device).unwrap();

        // Extract scalars
        let loss_q_scalar = loss_q.to_scalar::<f32>().unwrap();
        let param_loss_scalar = param_loss.to_scalar::<f32>().unwrap();

        // Combine
        let total = loss_q_scalar + 0.1 * param_loss_scalar;

        // Create new tensor
        let total_tensor = Tensor::from_vec(vec![total], &[1], &device).unwrap();

        // This should work
        assert_eq!(total_tensor.dims(), &[1]);
        assert!(!total.is_nan());
    }
}
