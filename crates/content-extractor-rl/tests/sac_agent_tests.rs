// ============================================================================
// FILE: crates/content-extractor-rl/tests/sac_agent_tests.rs
// ============================================================================

use content_extractor_rl::{Config, agents::{AgentFactory, AlgorithmType}};
use content_extractor_rl::replay_buffer::{PrioritizedReplayBuffer, Experience};
use candle_core::Device;
use tempfile::TempDir;
use std::collections::HashMap;

#[test]
fn test_sac_agent_creation() {
    let device = Device::Cpu;
    let config = Config::default();

    let agent = AgentFactory::create(
        AlgorithmType::SAC,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    assert_eq!(agent.algorithm_type(), AlgorithmType::SAC);

    let info = agent.get_info();
    assert!(info.features.contains(&"twin_q".to_string()));
    assert!(info.features.contains(&"entropy_regularization".to_string()));
    assert!(info.features.contains(&"automatic_temperature".to_string()));
}

#[test]
fn test_sac_action_selection() {
    let device = Device::Cpu;
    let config = Config::default();

    let agent = AgentFactory::create(
        AlgorithmType::SAC,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    // Test action selection
    let state = vec![0.5f32; config.state_dim];
    let result = agent.select_action(&state, 0.0);

    assert!(result.is_ok(), "Action selection failed");

    let (action, params) = result.unwrap();
    assert!(action < config.num_discrete_actions, "Invalid discrete action");
    assert_eq!(params.len(), config.num_continuous_params, "Wrong number of continuous params");

    // Params should be in valid range (tanh bounded)
    for &param in &params {
        assert!(param >= -1.0 && param <= 1.0, "Param out of range: {}", param);
    }
}

#[test]
fn test_sac_deterministic_inference() {
    let device = Device::Cpu;
    let config = Config::default();

    let agent = AgentFactory::create(
        AlgorithmType::SAC,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    let state = vec![0.75f32; config.state_dim];

    // Multiple calls should give same result (deterministic for inference)
    let results: Vec<_> = (0..5)
        .map(|_| agent.select_action(&state, 0.0).unwrap())
        .collect();

    // Check all actions are the same
    for i in 1..results.len() {
        assert_eq!(results[0].0, results[i].0, "Actions not deterministic");
        for j in 0..config.num_continuous_params {
            assert!((results[0].1[j] - results[i].1[j]).abs() < 1e-6,
                    "Params not deterministic");
        }
    }
}

#[test]
fn test_sac_training_step() {
    let device = Device::Cpu;
    let config = Config::default();

    let mut agent = AgentFactory::create(
        AlgorithmType::SAC,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    // Create replay buffer and fill it
    let mut replay_buffer = PrioritizedReplayBuffer::new(10000, 0.6, 0.4);

    for _ in 0..1000 {
        let exp = Experience {
            state: vec![0.5f32; config.state_dim],
            action: (0, vec![0.0f32; config.num_continuous_params]),
            reward: 1.0,
            next_state: vec![0.6f32; config.state_dim],
            done: false,
        };
        replay_buffer.add(exp);
    }

    // Perform training step
    let result = agent.train_step(&mut replay_buffer, 64);

    assert!(result.is_ok(), "Training step failed: {:?}", result);

    let loss = result.unwrap();
    assert!(!loss.is_nan(), "Loss is NaN");
    assert!(!loss.is_infinite(), "Loss is infinite");
    assert!(loss >= 0.0, "Loss is negative");

    // Step count should increment
    assert!(agent.get_step_count() > 0, "Step count not incremented");
}

#[test]
fn test_sac_save_formats() {
    let temp_dir = TempDir::new().unwrap();
    let device = Device::Cpu;
    let config = Config::default();

    let agent = AgentFactory::create(
        AlgorithmType::SAC,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    // Save in both formats
    let onnx_path = temp_dir.path().join("sac_model.onnx");
    let mut hyperparams = HashMap::new();
    hyperparams.insert("learning_rate".to_string(), 3e-4);
    hyperparams.insert("tau".to_string(), 0.005);

    agent.save_with_metadata(&onnx_path, 1000, hyperparams).unwrap();

    // Verify both files exist
    assert!(onnx_path.exists(), "ONNX file not created");

    let safetensors_path = onnx_path.with_extension("safetensors");
    assert!(safetensors_path.exists(), "SafeTensors file not created");

    // Verify file sizes are reasonable
    let onnx_size = std::fs::metadata(&onnx_path).unwrap().len();
    let safetensors_size = std::fs::metadata(&safetensors_path).unwrap().len();

    println!("SAC ONNX size: {} bytes", onnx_size);
    println!("SAC SafeTensors size: {} bytes", safetensors_size);

    assert!(onnx_size > 10_000, "ONNX file too small");
    assert!(safetensors_size > 10_000, "SafeTensors file too small");
}

#[test]
fn test_sac_entropy_learning() {
    let device = Device::Cpu;
    let config = Config::default();

    let mut agent = AgentFactory::create(
        AlgorithmType::SAC,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    // Create experiences with varied rewards
    let mut replay_buffer = PrioritizedReplayBuffer::new(10000, 0.6, 0.4);

    for i in 0..1000 {
        let reward = (i as f32 % 10.0) - 5.0; // Varied rewards
        let exp = Experience {
            state: vec![0.5f32; config.state_dim],
            action: (i % config.num_discrete_actions, vec![0.0f32; config.num_continuous_params]),
            reward,
            next_state: vec![0.6f32; config.state_dim],
            done: i % 100 == 0,
        };
        replay_buffer.add(exp);
    }

    // Perform multiple training steps
    let mut losses = Vec::new();
    for _ in 0..10 {
        if let Ok(loss) = agent.train_step(&mut replay_buffer, 64) {
            losses.push(loss);
        }
    }

    // Verify training is happening
    assert!(!losses.is_empty(), "No successful training steps");
    assert!(losses.iter().all(|&l| !l.is_nan()), "NaN losses detected");
    assert!(losses.iter().all(|&l| !l.is_infinite()), "Infinite losses detected");

    println!("SAC training losses: {:?}", losses);
}

#[test]
fn test_sac_continuous_action_bounds() {
    let device = Device::Cpu;
    let config = Config::default();

    let agent = AgentFactory::create(
        AlgorithmType::SAC,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    // Test with various states
    for val in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let state = vec![val; config.state_dim];
        let (_action, params) = agent.select_action(&state, 0.0).unwrap();
        // All continuous params should be bounded by tanh
        for &param in &params {
            assert!(param >= -1.0 && param <= 1.0,
                    "Param out of bounds: {} for state value {}", param, val);
        }
    }
}

#[test]
fn test_sac_load_and_inference() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("sac_test.onnx");
    let device = Device::Cpu;
    let config = Config::default();
    // Create and save
    let agent = AgentFactory::create(
        AlgorithmType::SAC,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    let state = vec![0.5f32; config.state_dim];
    let (action_before, params_before) = agent.select_action(&state, 0.0).unwrap();

    agent.save(&model_path).unwrap();

    // Load and test
    let loaded_agent = AgentFactory::load(
        &model_path,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        &device,
    ).unwrap();

    let (action_after, params_after) = loaded_agent.select_action(&state, 0.0).unwrap();

    // Should produce same outputs
    assert_eq!(action_before, action_after, "Discrete action changed");

    for (i, (before, after)) in params_before.iter().zip(params_after.iter()).enumerate() {
        assert!((before - after).abs() < 1e-5,
                "Continuous param {} changed: {} vs {}", i, before, after);
    }
}