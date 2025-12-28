// ============================================================================
// FILE: crates/article-extractor/tests/algorithm_save_load_tests.rs
// ============================================================================
use article_extractor::{Config, agents::{AgentFactory, AlgorithmType}};
use candle_core::Device;
use tempfile::TempDir;
use std::collections::HashMap;
#[test]
fn test_ppo_save_and_load() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("ppo_model.bin");
    let device = Device::Cpu;
    let config = Config::default();

    // Create PPO agent
    let mut agent = AgentFactory::create(
        AlgorithmType::PPO,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();
    // Get action before save
    let state = vec![0.5f32; config.state_dim];
    let (action_before, params_before) = agent.select_action(&state, 0.0).unwrap();

    // Save with metadata
    let mut hyperparams = HashMap::new();
    hyperparams.insert("learning_rate".to_string(), 3e-4);
    hyperparams.insert("clip_epsilon".to_string(), 0.2);
    agent.save_with_metadata(&model_path, 1000, hyperparams).unwrap();

    // Verify file exists
    assert!(model_path.exists());
    let file_size = std::fs::metadata(&model_path).unwrap().len();
    println!("PPO model size: {} bytes", file_size);
    assert!(file_size > 100_000, "Model file too small");

    // Load metadata
    let metadata = article_extractor::ModelMetadata::load_metadata(&model_path).unwrap();
    assert_eq!(metadata.algorithm, "PPO");
    assert_eq!(metadata.training_episodes, 1000);
    assert!(metadata.hyperparameters.contains_key("learning_rate"));

    // Load agent
    let loaded_agent = AgentFactory::load(
        &model_path,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        &device,
    ).unwrap();

    // Verify algorithm type
    assert_eq!(loaded_agent.algorithm_type(), AlgorithmType::PPO);

    // Get action after load - should be similar (PPO is stochastic)
    let (action_after, _params_after) = loaded_agent.select_action(&state, 0.0).unwrap();
    println!("Action before: {}, after: {}", action_before, action_after);
    // Note: PPO samples actions, so they might differ, but model should work
}

#[test]
fn test_sac_save_and_load() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("sac_model.bin");
    let device = Device::Cpu;
    let config = Config::default();

    // Create SAC agent
    let mut agent = AgentFactory::create(
        AlgorithmType::SAC,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    // Get action before save
    let state = vec![0.5f32; config.state_dim];
    let (action_before, params_before) = agent.select_action(&state, 0.0).unwrap();

    // Save with metadata
    let mut hyperparams = HashMap::new();
    hyperparams.insert("learning_rate".to_string(), 3e-4);
    hyperparams.insert("tau".to_string(), 0.005);
    agent.save_with_metadata(&model_path, 2000, hyperparams).unwrap();

    // Verify file exists
    assert!(model_path.exists());
    let file_size = std::fs::metadata(&model_path).unwrap().len();
    println!("SAC model size: {} bytes", file_size);
    assert!(file_size > 100_000, "Model file too small");

    // Load metadata
    let metadata = article_extractor::ModelMetadata::load_metadata(&model_path).unwrap();
    assert_eq!(metadata.algorithm, "SAC");
    assert_eq!(metadata.training_episodes, 2000);
    assert!(metadata.hyperparameters.contains_key("tau"));

    // Load agent
    let loaded_agent = AgentFactory::load(
        &model_path,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        &device,
    ).unwrap();

    // Verify algorithm type
    assert_eq!(loaded_agent.algorithm_type(), AlgorithmType::SAC);

    // Get action after load
    let (action_after, params_after) = loaded_agent.select_action(&state, 0.0).unwrap();

    // SAC uses deterministic policy for inference (mean), so should be same
    assert_eq!(action_before, action_after, "SAC action changed after load");

    for (i, (p_before, p_after)) in params_before.iter().zip(params_after.iter()).enumerate() {
        assert!(
            (p_before - p_after).abs() < 0.01,
            "SAC param {} changed after load: {} vs {}",
            i, p_before, p_after
        );
    }
}
#[test]
fn test_dqn_save_and_load_with_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("dqn_model.onnx");
    let device = Device::Cpu;
    let config = Config::default();

    // Create DQN agent
    let mut agent = AgentFactory::create(
        AlgorithmType::DuelingDQN,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    // Save with metadata
    let mut hyperparams = HashMap::new();
    hyperparams.insert("learning_rate".to_string(), 1e-3);
    hyperparams.insert("gamma".to_string(), 0.95);
    hyperparams.insert("epsilon_decay".to_string(), 0.995);
    agent.save_with_metadata(&model_path, 5000, hyperparams).unwrap();

    // Load metadata
    let metadata = article_extractor::ModelMetadata::load_metadata(&model_path).unwrap();
    assert_eq!(metadata.algorithm, "DuelingDQN");
    assert_eq!(metadata.training_episodes, 5000);
    assert_eq!(metadata.hyperparameters.get("gamma"), Some(&0.95));

    println!("DQN Metadata:");
    metadata.display();

    // Load agent
    let loaded_agent = AgentFactory::load(
        &model_path,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        &device,
    ).unwrap();

    assert_eq!(loaded_agent.algorithm_type(), AlgorithmType::DuelingDQN);
}
#[test]
fn test_cross_algorithm_detection() {
    let temp_dir = TempDir::new().unwrap();
    let device = Device::Cpu;
    let config = Config::default();
    // Create and save PPO
    let ppo_path = temp_dir.path().join("ppo.bin");
    let mut ppo_agent = AgentFactory::create(
        AlgorithmType::PPO,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();
    ppo_agent.save(&ppo_path).unwrap();

    // Create and save SAC
    let sac_path = temp_dir.path().join("sac.bin");
    let mut sac_agent = AgentFactory::create(
        AlgorithmType::SAC,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();
    sac_agent.save(&sac_path).unwrap();

    // Load and verify correct algorithm detection
    let loaded_ppo = AgentFactory::load(
        &ppo_path,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        &device,
    ).unwrap();
    assert_eq!(loaded_ppo.algorithm_type(), AlgorithmType::PPO);

    let loaded_sac = AgentFactory::load(
        &sac_path,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        &device,
    ).unwrap();
    assert_eq!(loaded_sac.algorithm_type(), AlgorithmType::SAC);
}
#[test]
fn test_metadata_persistence() {
    let temp_dir = TempDir::new().unwrap();
    let model_path = temp_dir.path().join("test_model.bin");
    let device = Device::Cpu;
    let config = Config::default();

    let mut agent = AgentFactory::create(
        AlgorithmType::PPO,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    // Create detailed metadata
    let mut hyperparams = HashMap::new();
    hyperparams.insert("learning_rate".to_string(), 3e-4);
    hyperparams.insert("clip_epsilon".to_string(), 0.2);
    hyperparams.insert("gae_lambda".to_string(), 0.95);
    hyperparams.insert("value_loss_coef".to_string(), 0.5);
    hyperparams.insert("entropy_coef".to_string(), 0.01);

    agent.save_with_metadata(&model_path, 10000, hyperparams.clone()).unwrap();

    // Load and verify all metadata
    let metadata = article_extractor::ModelMetadata::load_metadata(&model_path).unwrap();

    assert_eq!(metadata.algorithm, "PPO");
    assert_eq!(metadata.training_episodes, 10000);
    assert_eq!(metadata.state_dim, config.state_dim);
    assert_eq!(metadata.num_actions, config.num_discrete_actions);
    assert_eq!(metadata.num_params, config.num_continuous_params);

    // Verify all hyperparameters persisted
    for (key, value) in hyperparams.iter() {
        assert_eq!(
            metadata.hyperparameters.get(key),
            Some(value),
            "Hyperparameter {} not persisted correctly",
            key
        );
    }

    // Verify training date is set
    assert!(!metadata.training_date.is_empty());

    // Verify version is set
    assert_eq!(metadata.version, "1.0.0");
}
