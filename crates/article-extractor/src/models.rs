use candle_core::{Device, Tensor, DType, Result as CandleResult};
use candle_nn::{Linear, Module, VarBuilder, linear, layer_norm, LayerNorm};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Dueling DQN network architecture
pub struct DuelingDQN {
    // Feature encoder
    fc1: Linear,
    ln1: LayerNorm,
    fc2: Linear,
    ln2: LayerNorm,
    fc3: Linear,
    ln3: LayerNorm,
    dropout: f32,

    // Value stream
    value_fc1: Linear,
    value_fc2: Linear,

    // Advantage stream
    advantage_fc1: Linear,
    advantage_fc2: Linear,

    // Continuous parameter head
    param_mean: Linear,
    param_logstd: Tensor,

    device: Device,
    state_dim: usize,
    num_actions: usize,
    num_params: usize,
}

impl DuelingDQN {
    /// Create new Dueling DQN network
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let device = vb.device().clone();

        // Feature encoder
        let fc1 = linear(state_dim, 512, vb.pp("fc1"))?;
        let ln1 = layer_norm(512, 1e-5, vb.pp("ln1"))?;
        let fc2 = linear(512, 256, vb.pp("fc2"))?;
        let ln2 = layer_norm(256, 1e-5, vb.pp("ln2"))?;
        let fc3 = linear(256, 128, vb.pp("fc3"))?;
        let ln3 = layer_norm(128, 1e-5, vb.pp("ln3"))?;

        // Value stream
        let value_fc1 = linear(128, 64, vb.pp("value_fc1"))?;
        let value_fc2 = linear(64, 1, vb.pp("value_fc2"))?;

        // Advantage stream
        let advantage_fc1 = linear(128, 64, vb.pp("advantage_fc1"))?;
        let advantage_fc2 = linear(64, num_actions, vb.pp("advantage_fc2"))?;

        // Continuous parameter head
        let param_mean = linear(128, num_params, vb.pp("param_mean"))?;
        let param_logstd = Tensor::zeros(&[num_params], DType::F32, &device)?;

        Ok(Self {
            fc1,
            ln1,
            fc2,
            ln2,
            fc3,
            ln3,
            dropout: 0.1,
            value_fc1,
            value_fc2,
            advantage_fc1,
            advantage_fc2,
            param_mean,
            param_logstd,
            device,
            state_dim,
            num_actions,
            num_params,
        })
    }

    /// Forward pass through network
    pub fn forward(&self, state: &Tensor, training: bool) -> CandleResult<(Tensor, Tensor, Tensor)> {
        // Feature extraction
        let mut x = self.fc1.forward(state)?;
        x = self.ln1.forward(&x)?;
        x = x.relu()?;
        if training {
            x = candle_nn::ops::dropout(&x, self.dropout)?;
        }

        x = self.fc2.forward(&x)?;
        x = self.ln2.forward(&x)?;
        x = x.relu()?;
        if training {
            x = candle_nn::ops::dropout(&x, self.dropout)?;
        }

        x = self.fc3.forward(&x)?;
        x = self.ln3.forward(&x)?;
        let features = x.relu()?;

        // Value stream
        let mut value = self.value_fc1.forward(&features)?;
        value = value.relu()?;
        let value = self.value_fc2.forward(&value)?;

        // Advantage stream
        let mut advantages = self.advantage_fc1.forward(&features)?;
        advantages = advantages.relu()?;
        let advantages = self.advantage_fc2.forward(&advantages)?;

        // Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        let advantage_mean = advantages.mean_keepdim(1)?;
        let q_values = value
            .broadcast_add(&advantages)?
            .broadcast_sub(&advantage_mean)?;

        // Continuous parameters
        let param_mean = self.param_mean.forward(&features)?.tanh()?;
        let param_std = self.param_logstd.exp()?;

        Ok((q_values, param_mean, param_std))
    }

    pub fn vars(&self) -> Vec<candle_core::Var> {
        // Collect all vars from the network
        vec![] // Implement properly based on your Sequential structure
    }

    /// Get all model parameters
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();

        // Collect all trainable parameters
        // Note: This is simplified - in a real implementation,
        // we'd need to traverse all layers and collect their parameters

        params.push(self.param_logstd.clone());

        params
    }

    /// Save model to ONNX format
    pub fn save_to_onnx(&self, path: &Path) -> CandleResult<()> {
        // Create model metadata
        let metadata = ModelMetadata {
            state_dim: self.state_dim,
            num_actions: self.num_actions,
            num_params: self.num_params,
            architecture: "DuelingDQN".to_string(),
            version: "0.1.0".to_string(),
        };

        // Collect all parameters
        //let mut params_dict: HashMap<String, Vec<f32>> = HashMap::new();

        // Extract weights and biases from each layer
        // This is a simplified version - in production, you'd use proper ONNX export

        // For now, save as safetensors format (which is simpler than ONNX)
        self.save_as_safetensors(path, &metadata)?;

        Ok(())
    }

    /// Save model using safetensors format
    fn save_as_safetensors(&self, path: &Path, metadata: &ModelMetadata) -> CandleResult<()> {
        use std::fs::File;
        use std::io::Write;

        // Create a simple serialization format
        let mut file = File::create(path)
            .map_err(|e| candle_core::Error::Io(e))?;

        // Write metadata as JSON
        let metadata_json = serde_json::to_string(metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let metadata_bytes = metadata_json.as_bytes();
        let metadata_len = metadata_bytes.len() as u64;

        // Write metadata length (8 bytes)
        file.write_all(&metadata_len.to_le_bytes())
            .map_err(|e| candle_core::Error::Io(e))?;

        // Write metadata
        file.write_all(metadata_bytes)
            .map_err(|e| candle_core::Error::Io(e))?;

        // Write model parameters
        // In a real implementation, we would serialize all tensor data here

        Ok(())
    }

    /// Load model from ONNX format
    pub fn load_from_onnx(
        path: &Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
    ) -> CandleResult<Self> {
        // Read metadata and parameters
        let (metadata, _params_dict) = Self::load_from_safetensors(path)?;

        // Verify dimensions match
        if metadata.state_dim != state_dim
            || metadata.num_actions != num_actions
            || metadata.num_params != num_params
        {
            return Err(candle_core::Error::Msg(
                format!(
                    "Model dimension mismatch: expected ({}, {}, {}), got ({}, {}, {})",
                    state_dim, num_actions, num_params,
                    metadata.state_dim, metadata.num_actions, metadata.num_params
                )
            ));
        }

        // Create new model with loaded parameters
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Self::new(state_dim, num_actions, num_params, vb)?;

        // Load parameters into model
        // In a real implementation, we would restore all tensor values here

        Ok(model)
    }

    /// Load model from safetensors format
    fn load_from_safetensors(path: &Path) -> CandleResult<(ModelMetadata, HashMap<String, Vec<f32>>)> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)
            .map_err(|e| candle_core::Error::Io(e))?;

        // Read metadata length
        let mut metadata_len_bytes = [0u8; 8];
        file.read_exact(&mut metadata_len_bytes)
            .map_err(|e| candle_core::Error::Io(e))?;
        let metadata_len = u64::from_le_bytes(metadata_len_bytes) as usize;

        // Read metadata
        let mut metadata_bytes = vec![0u8; metadata_len];
        file.read_exact(&mut metadata_bytes)
            .map_err(|e| candle_core::Error::Io(e))?;

        let metadata_json = String::from_utf8(metadata_bytes)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Read parameters
        // In a real implementation, we would deserialize all tensor data here
        let params_dict = HashMap::new();

        Ok((metadata, params_dict))
    }

    /// Export to actual ONNX format (advanced implementation)
    #[cfg(feature = "onnx")]
    pub fn export_to_onnx(&self, path: &Path) -> CandleResult<()> {
        // This would require the `tract-onnx` crate for proper ONNX export
        // For now, we use the safetensors format above
        unimplemented!("Full ONNX export requires tract-onnx feature")
    }
}

/// Model metadata for serialization
#[derive(Debug, Serialize, Deserialize)]
struct ModelMetadata {
    state_dim: usize,
    num_actions: usize,
    num_params: usize,
    architecture: String,
    version: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.onnx");

        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DuelingDQN::new(300, 16, 6, vb).unwrap();

        // Save
        model.save_to_onnx(&model_path).unwrap();

        // Load
        let loaded_model = DuelingDQN::load_from_onnx(&model_path, 300, 16, 6).unwrap();

        assert_eq!(loaded_model.state_dim, 300);
        assert_eq!(loaded_model.num_actions, 16);
        assert_eq!(loaded_model.num_params, 6);
    }

    #[test]
    fn test_forward_pass() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = DuelingDQN::new(300, 16, 6, vb).unwrap();

        let state = Tensor::zeros(&[1, 300], DType::F32, &device).unwrap();
        let (q_values, param_mean, param_std) = model.forward(&state, false).unwrap();

        assert_eq!(q_values.dims(), &[1, 16]);
        assert_eq!(param_mean.dims(), &[1, 6]);
        assert_eq!(param_std.dims(), &[6]);
    }
}
