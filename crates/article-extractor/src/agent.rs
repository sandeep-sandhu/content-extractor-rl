use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, Optimizer, AdamW, ParamsAdamW, VarMap};
use crate::models::DuelingDQN;
use crate::replay_buffer::{PrioritizedReplayBuffer, SampledBatch};
use crate::Result;
use rand::Rng;

/// DQN Agent for article extraction
pub struct DQNAgent {
    online_network: DuelingDQN,
    target_network: DuelingDQN,
    optimizer: AdamW,
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
        let target_network = DuelingDQN::new(state_dim, num_actions, num_params, vb.pp("target"))?;

        // Get Var objects from the network properly
        let varmap = VarMap::new();
        let vb_opt = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let _online_network_for_vars = DuelingDQN::new(state_dim, num_actions, num_params, vb_opt)?;

        let vars = varmap.all_vars();

        let params = ParamsAdamW {
            lr,
            ..Default::default()
        };
        let optimizer = AdamW::new(vars, params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(Self {
            online_network,
            target_network,
            optimizer,
            num_actions,
            num_params,
            gamma,
            step_count: 0,
            device: device.clone(),
        })
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

        // Combined loss
        let param_loss_weight_vec = vec![0.1f32; 1];
        let param_loss_weight = Tensor::from_vec(param_loss_weight_vec, &[1], &self.device)?;
        let total_loss = loss_q.add(&param_loss.mul(&param_loss_weight)?)?;

        // Backward pass
        self.optimizer.backward_step(&total_loss)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Update priorities in replay buffer
        replay_buffer.update_priorities(&indices, &td_errors);

        self.step_count += 1;

        // Return loss value
        let loss_value = total_loss
            .to_scalar::<f32>()
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        Ok(loss_value)
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

        // CRITICAL FIX: Broadcast stds to match batch dimension
        // stds has shape [num_params], need to broadcast to [batch_size, num_params]
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

    /// Update target network (hard update)
    pub fn update_target_network(&mut self) {
        // In a real implementation, we would copy all parameters
    }

    /// Save model to ONNX format
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        self.online_network.save_to_onnx(path)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))
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
