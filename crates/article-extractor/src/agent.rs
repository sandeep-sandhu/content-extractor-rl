use candle_core::{Device, Tensor, DType, Result as CandleResult};
use candle_nn::{VarBuilder, Optimizer, AdamW, ParamsAdamW, ops::softmax, loss};
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
        learning_rate: f64,
        gamma: f32,
    ) -> Result<Self> {
        let device = Device::Cpu; // Use CPU for now, can be changed to CUDA

        // Create networks
        let vb_online = VarBuilder::zeros(DType::F32, &device);
        let online_network = DuelingDQN::new(state_dim, num_actions, num_params, vb_online)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let vb_target = VarBuilder::zeros(DType::F32, &device);
        let target_network = DuelingDQN::new(state_dim, num_actions, num_params, vb_target)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        // Create optimizer with parameters from online network
        let vars = online_network.parameters();
        let params = ParamsAdamW {
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
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
            device,
        })
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: &[f32], epsilon: f32) -> Result<(usize, Vec<f32>)> {
        let mut rng = rand::thread_rng();

        if rng.gen::<f32>() < epsilon {
            // Random action
            let discrete_action = rng.gen_range(0..self.num_actions);
            let continuous_params: Vec<f32> = (0..self.num_params)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();

            Ok((discrete_action, continuous_params))
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

        // Calculate TD targets
        let ones = Tensor::ones(&[batch_size], DType::F32, &self.device)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let discount_factors = (ones - dones_tensor)?
            .mul_scalar(self.gamma)?;

        let td_targets = (rewards_tensor + (next_q_values * discount_factors)?)?;

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
        let total_loss = (loss_q + param_loss.mul_scalar(0.1)?)?;

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
        param_means: &Tensor,
        param_stds: &Tensor,
        actions_params: &Tensor,
    ) -> CandleResult<Tensor> {
        // Negative log-likelihood of Gaussian distribution
        // -log(p(x)) = 0.5 * log(2π) + log(σ) + 0.5 * ((x - μ) / σ)²

        let diff = (actions_params - param_means)?;
        let normalized_diff = (diff / param_stds)?;
        let squared_diff = normalized_diff.sqr()?;

        let log_std = param_stds.log()?;
        let log_2pi = std::f32::consts::PI * 2.0;
        let constant = Tensor::new(&[log_2pi.ln() * 0.5], &self.device)?;

        let nll = (constant + log_std + squared_diff.mul_scalar(0.5)?)?;

        nll.mean_all()
    }

    /// Update target network (hard update)
    pub fn update_target_network(&mut self) {
        // In a real implementation, we would copy all parameters
        // For now, this is a placeholder
        // self.target_network.load_state_dict(self.online_network.state_dict());
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
        let device = Device::Cpu;

        let online_network = DuelingDQN::load_from_onnx(path, state_dim, num_actions, num_params)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let vb_target = VarBuilder::zeros(DType::F32, &device);
        let target_network = DuelingDQN::new(state_dim, num_actions, num_params, vb_target)
            .map_err(|e| crate::ExtractionError::ModelError(e.to_string()))?;

        let vars = online_network.parameters();
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
}

/// Smooth L1 loss (Huber loss)
fn smooth_l1_loss(predictions: &Tensor, targets: &Tensor) -> CandleResult<Tensor> {
    let diff = (predictions - targets)?.abs()?;

    // If |diff| < 1: 0.5 * diff²
    // Else: |diff| - 0.5
    let mask = diff.lt(1.0)?;

    let small_loss = diff.sqr()?.mul_scalar(0.5)?;
    let large_loss = (diff - 0.5)?;

    mask.where_cond(&small_loss, &large_loss)
}
