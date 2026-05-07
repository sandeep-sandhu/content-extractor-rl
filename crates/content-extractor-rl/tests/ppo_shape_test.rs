// ============================================================================
// TEST: Verify PPO shape fixes
// FILE: crates/content-extractor-rl/tests/ppo_shape_test.rs
// ============================================================================

use content_extractor_rl::{Config, agents::{AgentFactory, AlgorithmType}};
use candle_core::Device;
use content_extractor_rl::replay_buffer::{PrioritizedReplayBuffer, Experience};

#[test]
fn test_ppo_advantage_normalization_small_batch() {
    let device = Device::Cpu;
    let config = Config::default();
    let _varmap = candle_nn::VarMap::new();

    let mut agent = AgentFactory::create(
        AlgorithmType::PPO,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    let mut replay_buffer = PrioritizedReplayBuffer::new(10000, 0.6, 0.4);

    // Small batch
    for _ in 0..64 {
        let exp = Experience {
            state: vec![0.1; 300],
            action: (0, vec![0.0; 6]),
            reward: 1.0,
            next_state: vec![0.2; 300],
            done: false,
        };
        replay_buffer.add(exp);
    }

    let result = agent.train_step(&mut replay_buffer, 32);
    assert!(result.is_ok(), "PPO training failed with batch_size=32: {:?}", result.err());
}

#[test]
fn test_ppo_advantage_normalization_large_batch() {
    let device = Device::Cpu;
    let config = Config::default();
    let varmap = candle_nn::VarMap::new();

    let mut agent = AgentFactory::create(
        AlgorithmType::PPO,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    ).unwrap();

    let mut replay_buffer = PrioritizedReplayBuffer::new(10000, 0.6, 0.4);

    // Large batch (the failing case)
    for _ in 0..1024 {
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
    assert!(result.is_ok(), "PPO training failed with batch_size=512: {:?}", result.err());
}

#[test]
fn test_ppo_various_batch_sizes() {
    let device = Device::Cpu;
    let config = Config::default();

    for batch_size in [1, 16, 32, 64, 128, 256, 512, 1024] {
        let mut agent = AgentFactory::create(
            AlgorithmType::PPO,
            config.state_dim,
            config.num_discrete_actions,
            config.num_continuous_params,
            config.gamma as f32,
            config.learning_rate,
            &device,
        ).unwrap();

        let mut replay_buffer = PrioritizedReplayBuffer::new(10000, 0.6, 0.4);

        for _ in 0..(batch_size * 2) {
            let exp = Experience {
                state: vec![0.1; 300],
                action: (0, vec![0.0; 6]),
                reward: 1.0,
                next_state: vec![0.2; 300],
                done: false,
            };
            replay_buffer.add(exp);
        }

        let result = agent.train_step(&mut replay_buffer, batch_size);
        assert!(
            result.is_ok(),
            "PPO training failed with batch_size={}: {:?}",
            batch_size,
            result.err()
        );
    }
}

#[test]
fn test_ppo_entropy_calculation() {
    use candle_core::Tensor;

    let device = Device::Cpu;

    // Test with different num_params
    for num_params in [1, 3, 6, 12] {
        let std = Tensor::rand(0.0, 1.0, &[num_params], &device).unwrap();
        let logits = Tensor::rand(0.0, 1.0, &[32, 16], &device).unwrap();

        // This should not panic with shape errors
        let entropy_result = calculate_entropy_helper(&logits, &std);
        assert!(
            entropy_result.is_ok(),
            "Entropy calculation failed with num_params={}: {:?}",
            num_params,
            entropy_result.err()
        );
    }
}

// Helper function for testing (you might need to expose this or make it public)
fn calculate_entropy_helper(
    logits: &candle_core::Tensor,
    std: &candle_core::Tensor,
) -> candle_core::error::Result<candle_core::Tensor> {
    use candle_nn::ops::{softmax, log_softmax};

    // Discrete entropy
    let probs = softmax(&logits, 1)?;
    let log_probs = log_softmax(&logits, 1)?;
    let discrete_entropy = -1.0 * (probs * log_probs)?.sum(1)?.mean_all()?;

    // Continuous entropy
    let num_params = std.dims()[0];
    let constant = candle_core::Tensor::new(
        vec![0.5_f64 * (1.0_f64 + 2.0_f64 * std::f64::consts::PI).ln(); num_params],
        std.device()
    )?;

    let continuous_entropy = (std.log()? + constant)?.mean_all()?;

    discrete_entropy + continuous_entropy
}