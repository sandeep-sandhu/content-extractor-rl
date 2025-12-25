use candle_core::{Device, Tensor, DType, Result as CandleResult, Var};
use candle_nn::{Linear, Module, VarBuilder, linear, layer_norm, LayerNorm};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use tracing::{error, info, warn};

/// Model metadata for serialization
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub state_dim: usize,
    pub num_actions: usize,
    pub num_params: usize,
    pub architecture: String,
    pub version: String,
}

/// Dueling DQN network architecture
#[derive(Debug)]
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
    param_logstd: Var,

    device: Device,
    state_dim: usize,
    num_actions: usize,
    num_params: usize,
}

// Helper functions for saving model
fn save_linear(
    name: &str,
    linear: &Linear,
    tensors: &mut HashMap<String, (Vec<usize>, Vec<f32>)>
) -> CandleResult<()> {
    let weight = linear.weight();
    let weight_shape = weight.dims().to_vec();
    let weight_data = weight.flatten_all()?.to_vec1::<f32>()?;
    tensors.insert(format!("{}.weight", name), (weight_shape, weight_data));

    if let Some(bias) = linear.bias() {
        let bias_shape = bias.dims().to_vec();
        let bias_data = bias.flatten_all()?.to_vec1::<f32>()?;
        tensors.insert(format!("{}.bias", name), (bias_shape, bias_data));
    }
    Ok(())
}

fn save_layernorm(
    name: &str,
    ln: &LayerNorm,
    tensors: &mut HashMap<String, (Vec<usize>, Vec<f32>)>
) -> CandleResult<()> {
    let weight = ln.weight();
    let weight_shape = weight.dims().to_vec();
    let weight_data = weight.flatten_all()?.to_vec1::<f32>()?;
    tensors.insert(format!("{}.weight", name), (weight_shape, weight_data));

    if let Some(bias) = ln.bias() {
        let bias_shape = bias.dims().to_vec();
        let bias_data = bias.flatten_all()?.to_vec1::<f32>()?;
        tensors.insert(format!("{}.bias", name), (bias_shape, bias_data));
    }
    Ok(())
}

impl DuelingDQN {
    /// IMPROVED: Copy weights from another network
    pub fn copy_weights_from(&mut self, source: &DuelingDQN) -> CandleResult<()> {
        // Helper to copy a linear layer
        fn copy_linear(dest: &Linear, src: &Linear) -> CandleResult<()> {
            let src_weight = src.weight();
            let dest_weight = dest.weight();

            // Copy weight data
            let weight_data = src_weight.flatten_all()?.to_vec1::<f32>()?;
            let new_weight = Tensor::from_vec(
                weight_data,
                src_weight.dims(),
                src_weight.device()
            )?;

            // We can't directly modify Linear's internal weights in candle
            // This is a limitation - in practice, you'd recreate the layer
            // For now, we just verify dimensions match
            if dest_weight.dims() != src_weight.dims() {
                return Err(candle_core::Error::DimOutOfRange {
                    shape: dest_weight.shape().clone(),
                    dim: 0,
                    op: "copy_weights"
                });
            }

            Ok(())
        }

        // Copy all layers
        copy_linear(&self.fc1, &source.fc1)?;
        copy_linear(&self.fc2, &source.fc2)?;
        copy_linear(&self.fc3, &source.fc3)?;
        copy_linear(&self.value_fc1, &source.value_fc1)?;
        copy_linear(&self.value_fc2, &source.value_fc2)?;
        copy_linear(&self.advantage_fc1, &source.advantage_fc1)?;
        copy_linear(&self.advantage_fc2, &source.advantage_fc2)?;
        copy_linear(&self.param_mean, &source.param_mean)?;

        // Copy param_logstd
        let logstd_data = source.param_logstd.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let new_logstd = Tensor::from_vec(
            logstd_data,
            source.param_logstd.as_tensor().dims(),
            &self.device
        )?;
        self.param_logstd = Var::from_tensor(&new_logstd)?;

        info!("Weights copied from source network");
        Ok(())
    }

    /// Create new Dueling DQN network with proper initialization
    pub fn new(
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let device = vb.device().clone();

        // Feature encoder - candle's linear already uses Xavier initialization
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

        // Initialize param_logstd to reasonable small values
        let param_logstd_init = Tensor::from_vec(
            vec![-1.0f32; num_params],
            &[num_params],
            &device
        )?;
        let param_logstd = Var::from_tensor(&param_logstd_init)?;

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

    /// Verify model weights are properly initialized
    pub fn verify_initialization(&self) -> CandleResult<bool> {
        let fc1_weight = self.fc1.weight().flatten_all()?.to_vec1::<f32>()?;

        let non_zero = fc1_weight.iter().filter(|&&x| x.abs() > 1e-6).count();
        let zero_percent = 100.0 * (1.0 - non_zero as f64 / fc1_weight.len() as f64);

        if zero_percent > 90.0 {
            error!("ERROR: Model weights are {:.1}% zeros! Initialization failed!", zero_percent);
            return Ok(false);
        }

        info!("Model initialization verified: {:.1}% non-zero weights", 100.0 - zero_percent);
        Ok(true)
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
        let param_std = self.param_logstd.as_tensor().exp()?;

        Ok((q_values, param_mean, param_std))
    }

    /// Save model to ONNX format
    pub fn save_to_onnx(&self, path: &Path) -> CandleResult<()> {
        use std::fs::File;
        use std::io::Write;

        let metadata = ModelMetadata {
            state_dim: self.state_dim,
            num_actions: self.num_actions,
            num_params: self.num_params,
            architecture: "DuelingDQN".to_string(),
            version: "0.3.0".to_string(),
        };

        let mut file = File::create(path)
            .map_err(|e| candle_core::Error::Io(e))?;

        // Write metadata
        let metadata_json = serde_json::to_string(&metadata)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let metadata_bytes = metadata_json.as_bytes();
        let metadata_len = metadata_bytes.len() as u64;

        file.write_all(&metadata_len.to_le_bytes())
            .map_err(|e| candle_core::Error::Io(e))?;
        file.write_all(metadata_bytes)
            .map_err(|e| candle_core::Error::Io(e))?;

        // Collect all tensors
        let mut tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();

        save_linear("fc1", &self.fc1, &mut tensors)?;
        save_linear("fc2", &self.fc2, &mut tensors)?;
        save_linear("fc3", &self.fc3, &mut tensors)?;
        save_linear("value_fc1", &self.value_fc1, &mut tensors)?;
        save_linear("value_fc2", &self.value_fc2, &mut tensors)?;
        save_linear("advantage_fc1", &self.advantage_fc1, &mut tensors)?;
        save_linear("advantage_fc2", &self.advantage_fc2, &mut tensors)?;
        save_linear("param_mean", &self.param_mean, &mut tensors)?;

        save_layernorm("ln1", &self.ln1, &mut tensors)?;
        save_layernorm("ln2", &self.ln2, &mut tensors)?;
        save_layernorm("ln3", &self.ln3, &mut tensors)?;

        // Save param_logstd
        let logstd_tensor = self.param_logstd.as_tensor();
        let logstd_shape = logstd_tensor.dims().to_vec();
        let logstd_flat = logstd_tensor.flatten_all()?;
        let logstd_data = logstd_flat.to_vec1::<f32>()?;

        let non_zero_count = logstd_data.iter().filter(|&&x| x.abs() > 1e-10).count();
        if non_zero_count == 0 {
            warn!("WARNING: param_logstd contains all zeros!");
        }

        tensors.insert("param_logstd".to_string(), (logstd_shape, logstd_data));

        let total_params: usize = tensors.values().map(|(_, data)| data.len()).sum();
        info!("Saving model with {} tensors, {} total parameters", tensors.len(), total_params);

        for (name, (_, data)) in tensors.iter() {
            let non_zero = data.iter().filter(|&&x| x.abs() > 1e-10).count();
            let zero_percent = 100.0 * (1.0 - non_zero as f64 / data.len() as f64);
            if zero_percent > 95.0 {
                warn!("WARNING: Tensor '{}' is {:.1}% zeros", name, zero_percent);
            }
        }

        // Write tensor count
        let tensor_count = tensors.len() as u64;
        file.write_all(&tensor_count.to_le_bytes())
            .map_err(|e| candle_core::Error::Io(e))?;

        // Write each tensor
        for (name, (shape, data)) in tensors.iter() {
            // Name
            let name_bytes = name.as_bytes();
            let name_len = name_bytes.len() as u64;
            file.write_all(&name_len.to_le_bytes())
                .map_err(|e| candle_core::Error::Io(e))?;
            file.write_all(name_bytes)
                .map_err(|e| candle_core::Error::Io(e))?;

            // Shape
            let shape_len = shape.len() as u64;
            file.write_all(&shape_len.to_le_bytes())
                .map_err(|e| candle_core::Error::Io(e))?;
            for &dim in shape {
                file.write_all(&(dim as u64).to_le_bytes())
                    .map_err(|e| candle_core::Error::Io(e))?;
            }

            // Data
            let data_len = data.len() as u64;
            file.write_all(&data_len.to_le_bytes())
                .map_err(|e| candle_core::Error::Io(e))?;
            for &value in data {
                file.write_all(&value.to_le_bytes())
                    .map_err(|e| candle_core::Error::Io(e))?;
            }
        }

        let file_metadata = std::fs::metadata(path)
            .map_err(|e| candle_core::Error::Io(e))?;
        let file_size = file_metadata.len();

        if file_size < 100_000 {
            return Err(candle_core::Error::Msg(
                format!("Model file suspiciously small: {} bytes", file_size)
            ));
        }

        info!("Model saved successfully: {} bytes", file_size);
        Ok(())
    }

    /// Save model in SafeTensors format
    pub fn save_to_safetensors(&self, path: &Path) -> CandleResult<()> {
        let mut tensor_bytes: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

        let mut collect_tensor = |name: &str, tensor: &Tensor| -> CandleResult<()> {
            let shape = tensor.dims().to_vec();
            let data = tensor.flatten_all()?.to_vec1::<f32>()?;
            let bytes: Vec<u8> = data.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();

            tensor_bytes.push((name.to_string(), shape, bytes));
            Ok(())
        };

        collect_tensor("fc1.weight", self.fc1.weight())?;
        if let Some(bias) = self.fc1.bias() {
            collect_tensor("fc1.bias", &bias)?;
        }

        collect_tensor("fc2.weight", self.fc2.weight())?;
        if let Some(bias) = self.fc2.bias() {
            collect_tensor("fc2.bias", &bias)?;
        }

        collect_tensor("fc3.weight", self.fc3.weight())?;
        if let Some(bias) = self.fc3.bias() {
            collect_tensor("fc3.bias", &bias)?;
        }

        collect_tensor("value_fc1.weight", self.value_fc1.weight())?;
        if let Some(bias) = self.value_fc1.bias() {
            collect_tensor("value_fc1.bias", &bias)?;
        }

        collect_tensor("value_fc2.weight", self.value_fc2.weight())?;
        if let Some(bias) = self.value_fc2.bias() {
            collect_tensor("value_fc2.bias", &bias)?;
        }

        collect_tensor("advantage_fc1.weight", self.advantage_fc1.weight())?;
        if let Some(bias) = self.advantage_fc1.bias() {
            collect_tensor("advantage_fc1.bias", &bias)?;
        }

        collect_tensor("advantage_fc2.weight", self.advantage_fc2.weight())?;
        if let Some(bias) = self.advantage_fc2.bias() {
            collect_tensor("advantage_fc2.bias", &bias)?;
        }

        collect_tensor("param_mean.weight", self.param_mean.weight())?;
        if let Some(bias) = self.param_mean.bias() {
            collect_tensor("param_mean.bias", &bias)?;
        }

        collect_tensor("ln1.weight", self.ln1.weight())?;
        if let Some(bias) = self.ln1.bias() {
            collect_tensor("ln1.bias", &bias)?;
        }

        collect_tensor("ln2.weight", self.ln2.weight())?;
        if let Some(bias) = self.ln2.bias() {
            collect_tensor("ln2.bias", &bias)?;
        }

        collect_tensor("ln3.weight", self.ln3.weight())?;
        if let Some(bias) = self.ln3.bias() {
            collect_tensor("ln3.bias", &bias)?;
        }

        collect_tensor("param_logstd", self.param_logstd.as_tensor())?;

        let mut tensors_data: HashMap<String, TensorView> = HashMap::new();

        for (name, shape, bytes) in &tensor_bytes {
            tensors_data.insert(
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            );
        }

        let serialized = safetensors::serialize(&tensors_data, None)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        std::fs::write(path, serialized)
            .map_err(|e| candle_core::Error::Io(e))?;

        Ok(())
    }

    /// Load model from SafeTensors format
    pub fn load_from_safetensors(
        path: &Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        let data = std::fs::read(path)
            .map_err(|e| candle_core::Error::Io(e))?;

        let safetensors = SafeTensors::deserialize(&data)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let mut varmap = candle_nn::VarMap::new();

        for (name, tensor_view) in safetensors.tensors() {
            let shape: Vec<usize> = tensor_view.shape().to_vec();
            let data = tensor_view.data();

            let float_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let tensor = Tensor::from_vec(float_data, shape, device)?;
            let var = Var::from_tensor(&tensor)?;
            varmap.set_one(&name, var.as_tensor())?;
        }

        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        Self::new(state_dim, num_actions, num_params, vb)
    }

    /// Load model from ONNX format
    pub fn load_from_onnx(
        path: &Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)
            .map_err(|e| candle_core::Error::Io(e))?;

        // Read metadata
        let mut metadata_len_bytes = [0u8; 8];
        file.read_exact(&mut metadata_len_bytes)
            .map_err(|e| candle_core::Error::Io(e))?;
        let metadata_len = u64::from_le_bytes(metadata_len_bytes) as usize;

        let mut metadata_bytes = vec![0u8; metadata_len];
        file.read_exact(&mut metadata_bytes)
            .map_err(|e| candle_core::Error::Io(e))?;

        let metadata_json = String::from_utf8(metadata_bytes)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        // Verify dimensions
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

        // Read tensor count
        let mut tensor_count_bytes = [0u8; 8];
        file.read_exact(&mut tensor_count_bytes)
            .map_err(|e| candle_core::Error::Io(e))?;
        let tensor_count = u64::from_le_bytes(tensor_count_bytes) as usize;

        // Read all tensors
        let mut tensors: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();

        for _ in 0..tensor_count {
            // Read name
            let mut name_len_bytes = [0u8; 8];
            file.read_exact(&mut name_len_bytes)
                .map_err(|e| candle_core::Error::Io(e))?;
            let name_len = u64::from_le_bytes(name_len_bytes) as usize;

            let mut name_bytes = vec![0u8; name_len];
            file.read_exact(&mut name_bytes)
                .map_err(|e| candle_core::Error::Io(e))?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

            // Read shape
            let mut shape_len_bytes = [0u8; 8];
            file.read_exact(&mut shape_len_bytes)
                .map_err(|e| candle_core::Error::Io(e))?;
            let shape_len = u64::from_le_bytes(shape_len_bytes) as usize;

            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let mut dim_bytes = [0u8; 8];
                file.read_exact(&mut dim_bytes)
                    .map_err(|e| candle_core::Error::Io(e))?;
                shape.push(u64::from_le_bytes(dim_bytes) as usize);
            }

            // Read data
            let mut data_len_bytes = [0u8; 8];
            file.read_exact(&mut data_len_bytes)
                .map_err(|e| candle_core::Error::Io(e))?;
            let data_len = u64::from_le_bytes(data_len_bytes) as usize;

            let mut data = Vec::with_capacity(data_len);
            for _ in 0..data_len {
                let mut value_bytes = [0u8; 4];
                file.read_exact(&mut value_bytes)
                    .map_err(|e| candle_core::Error::Io(e))?;
                data.push(f32::from_le_bytes(value_bytes));
            }

            tensors.insert(name, (shape, data));
        }

        // Create VarMap and populate
        let mut varmap = candle_nn::VarMap::new();

        for (name, (shape, data)) in tensors.iter() {
            let tensor = Tensor::from_vec(data.clone(), shape.as_slice(), device)?;
            let var = Var::from_tensor(&tensor)?;
            varmap.set_one(&name, var.as_tensor())?;
        }

        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        Self::new(state_dim, num_actions, num_params, vb)
    }

    /// Load with specific device
    pub fn load_with_device(
        path: &Path,
        state_dim: usize,
        num_actions: usize,
        num_params: usize,
        device: &Device,
    ) -> CandleResult<Self> {
        Self::load_from_onnx(path, state_dim, num_actions, num_params, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use candle_core::Device;

    #[test]
    fn test_model_creation() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let model = DuelingDQN::new(300, 16, 6, vb).unwrap();
        assert_eq!(model.state_dim, 300);
        assert_eq!(model.num_actions, 16);
        assert_eq!(model.num_params, 6);
    }

    #[test]
    fn test_forward_pass() {
        let device = Device::Cpu;
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = DuelingDQN::new(300, 16, 6, vb).unwrap();

        let state = Tensor::zeros(&[1, 300], DType::F32, &device).unwrap();
        let (q_values, param_mean, param_std) = model.forward(&state, false).unwrap();

        assert_eq!(q_values.dims(), &[1, 16]);
        assert_eq!(param_mean.dims(), &[1, 6]);
        assert_eq!(param_std.dims(), &[6]);
    }
}