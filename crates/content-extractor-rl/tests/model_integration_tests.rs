//! Comprehensive integration tests for model saving and loading
// ============================================================================
// FILE: crates/content-extractor-rl/tests/model_integration_tests.rs
// ============================================================================
use content_extractor_rl::agents::dqn_agent::DQNAgent;

#[cfg(test)]
mod integration_tests {
    use content_extractor_rl::{Config, Result};
    use candle_core::Device;
    use tempfile::TempDir;
    use std::path::PathBuf;
    use content_extractor_rl::agents::dqn_agent::DQNAgent;
    use content_extractor_rl::models::NetworkConfig;

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
    fn test_agent_save_load_full_cycle() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("agent_model.onnx");

        let device = Device::Cpu;
        let config = Config::default();
        let network_config = create_network_config(&config);

        // Create agent
        let agent = DQNAgent::new(
            network_config,
            config.gamma as f32,
            config.learning_rate,
            &device,
            candle_nn::VarMap::new(),
        ).unwrap();

        // Test action selection before save
        let state = vec![0.5f32; config.state_dim];
        let (action_before, params_before) = agent.select_action(&state, 0.0).unwrap();

        // Save
        agent.save(&model_path).unwrap();

        // Verify file size
        let metadata = std::fs::metadata(&model_path).unwrap();
        println!("Saved agent model: {} bytes", metadata.len());
        assert!(metadata.len() > 1_000_000, "Agent model file too small");

        // Load
        let loaded_agent = DQNAgent::load(
            &model_path,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
        ).unwrap();

        // Test action selection after load (should be same with epsilon=0)
        let (action_after, params_after) = loaded_agent.select_action(&state, 0.0).unwrap();

        assert_eq!(action_before, action_after, "Action changed after load");

        for (p_before, p_after) in params_before.iter().zip(params_after.iter()) {
            assert!((p_before - p_after).abs() < 1e-4,
                    "Parameter changed after load: {} vs {}", p_before, p_after);
        }
    }

    #[test]
    fn test_training_checkpoint_resume() {
        // This test verifies that training can be resumed from a checkpoint
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_path = temp_dir.path().join("checkpoint.onnx");

        let device = Device::Cpu;
        let config = Config::default();
        let network_config = create_network_config(&config);

        // Create and "train" agent
        let mut agent = DQNAgent::new(
            network_config,
            config.gamma as f32,
            config.learning_rate,
            &device,
            candle_nn::VarMap::new(),
        ).unwrap();

        // Simulate some training (just to modify weights)
        let state = vec![0.5f32; config.state_dim];
        let (action, _) = agent.select_action(&state, 0.1).unwrap();

        let step_count_before = agent.get_step_count();

        // Save checkpoint
        agent.save(&checkpoint_path).unwrap();

        // Simulate program restart - load from checkpoint
        let resumed_agent = DQNAgent::load(
            &checkpoint_path,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
        ).unwrap();

        // Verify state is preserved
        assert_eq!(resumed_agent.get_step_count(), step_count_before);
    }

    #[test]
    fn test_multiple_models_different_sizes() {
        let temp_dir = TempDir::new().unwrap();

        let device = Device::Cpu;

        // Test different model sizes
        let configs = vec![
            (50, 4, 2),   // Small model
            (300, 16, 6), // Standard model
            (500, 32, 10), // Large model
        ];

        for (state_dim, num_actions, num_params) in configs {
            let model_path = temp_dir.path().join(format!("model_{}_{}.onnx", state_dim, num_actions));

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

            let agent = DQNAgent::new(
                network_config,
                0.95,
                0.001,
                &device,
                candle_nn::VarMap::new(),
            ).unwrap();

            // Save
            agent.save(&model_path).unwrap();

            // Check file exists and has reasonable size
            let metadata = std::fs::metadata(&model_path).unwrap();
            println!("Model ({}, {}, {}): {} bytes",
                     state_dim, num_actions, num_params, metadata.len());

            // Larger models should have larger files
            let expected_min_size = (state_dim * 512 * 4) as u64; // Just first layer
            assert!(metadata.len() > expected_min_size,
                    "Model file smaller than expected for dimensions ({}, {}, {})",
                    state_dim, num_actions, num_params);

            // Load and verify
            let loaded = DQNAgent::load(
                &model_path,
                state_dim,
                num_actions,
                num_params,
            ).unwrap();

            // Test forward pass works
            let state = vec![0.5f32; state_dim];
            let result = loaded.select_action(&state, 0.0);
            assert!(result.is_ok(), "Forward pass failed after load");
        }
    }

    #[test]
    fn test_corrupted_file_handling() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("corrupted.onnx");

        // Create a corrupted file
        std::fs::write(&model_path, b"corrupted data").unwrap();

        let config = Config::default();

        // Try to load - should fail gracefully
        let result = DQNAgent::load(
            &model_path,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
        );

        assert!(result.is_err(), "Should fail to load corrupted file");
    }

    #[test]
    fn test_file_not_found_handling() {
        let config = Config::default();

        let result = DQNAgent::load(
            &PathBuf::from("/nonexistent/path/model.onnx"),
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
        );

        assert!(result.is_err(), "Should fail when file doesn't exist");
    }

    #[test]
    fn test_cross_device_compatibility() {
        // Test saving on CPU and loading on CPU (GPU test would require GPU)
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("cross_device.onnx");

        let device_save = Device::Cpu;
        let config = Config::default();
        let network_config = create_network_config(&config);

        // Save on CPU
        let agent = DQNAgent::new(
            network_config,
            config.gamma as f32,
            config.learning_rate,
            &device_save,
            candle_nn::VarMap::new(),
        ).unwrap();

        agent.save(&model_path).unwrap();

        // Load on CPU (in real scenario, try GPU if available)
        let device_load = Device::Cpu;
        let loaded = DQNAgent::load(
            &model_path,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
        ).unwrap();

        // Verify forward pass works
        let state = vec![0.5f32; config.state_dim];
        let result = loaded.select_action(&state, 0.0);
        assert!(result.is_ok(), "Forward pass failed after cross-device load");
    }

    #[test]
    fn test_deterministic_output_after_load() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("deterministic.onnx");

        let device = Device::Cpu;
        let config = Config::default();
        let network_config = create_network_config(&config);

        let agent = DQNAgent::new(
            network_config,
            config.gamma as f32,
            config.learning_rate,
            &device,
            candle_nn::VarMap::new(),
        ).unwrap();

        let state = vec![0.75f32; config.state_dim];

        // Get outputs before save (epsilon=0 for deterministic)
        let outputs_before: Vec<_> = (0..10)
            .map(|_| agent.select_action(&state, 0.0).unwrap())
            .collect();

        // Save and load
        agent.save(&model_path).unwrap();
        let loaded = DQNAgent::load(
            &model_path,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
        ).unwrap();

        // Get outputs after load
        let outputs_after: Vec<_> = (0..10)
            .map(|_| loaded.select_action(&state, 0.0).unwrap())
            .collect();

        // All outputs should be identical
        for (i, (before, after)) in outputs_before.iter().zip(outputs_after.iter()).enumerate() {
            assert_eq!(before.0, after.0, "Action {} mismatch", i);
            for (j, (p_before, p_after)) in before.1.iter().zip(after.1.iter()).enumerate() {
                assert!((p_before - p_after).abs() < 1e-6,
                        "Param {} of output {} changed: {} vs {}", j, i, p_before, p_after);
            }
        }
    }
}