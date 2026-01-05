// ============================================================================
// FILE: crates/article-extractor/src/training.rs
// ============================================================================

use crate::{
    Config, ArticleExtractionEnvironment, BaselineExtractor, ExtractionError,
    agents::{AgentFactory, RLAgent},
};

use crate::{
    replay_buffer::PrioritizedReplayBuffer,
    SiteProfileMemory,
    reward::ImprovedRewardCalculator,
    curriculum::CurriculumManager,
    Result,
};

use crate::environment::StepInfo;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tracing::{info, warn};
use crate::{Checkpoint, CheckpointManager};
use candle_nn::{VarMap};
use candle_core::Device;


/// Extract domain from URL
/// The second element of html_samples is now the actual URL from JSON
fn extract_domain_from_url(url: &str) -> String {
    use url::Url;

    // Parse the URL to extract domain
    match Url::parse(url) {
        Ok(parsed_url) => {
            parsed_url.host_str()
                .map(|h| h.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        }
        Err(_) => {
            // If URL parsing fails, try to extract domain directly
            let url = url.trim();
            let without_protocol = if url.starts_with("https://") {
                &url[8..]
            } else if url.starts_with("http://") {
                &url[7..]
            } else {
                url
            };

            // Split by '/' to get the host part
            let host_part = without_protocol.split('/').next().unwrap_or("");

            // Split by ':' to remove port (if any)
            let domain = host_part.split(':').next().unwrap_or("");

            if domain.is_empty() {
                "unknown".to_string()
            } else {
                domain.to_string()
            }
        }
    }
}


/// Training metrics
#[derive(Debug, Clone, Default)]  // Add Default derive
pub struct TrainingMetrics {
    pub episode_rewards: Vec<f32>,
    pub episode_qualities: Vec<f32>,
    pub episode_losses: Vec<f32>,
    pub best_avg_quality: f32,
}


/// Standard training loop with checkpoint support
pub fn train_standard(
    config: &Config,
    html_samples: Vec<(String, String)>,
) -> Result<(Box<dyn RLAgent>, TrainingMetrics)> {
    info!("Starting standard training for {} episodes", config.num_episodes);

    let device = if config.use_cpu_for_tuning {
        Device::Cpu
    } else if crate::cuda_is_available() {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    let varmap = VarMap::new();

    // Initialize components
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());
    let mut site_memory = SiteProfileMemory::new(&config.site_profiles_dir)?;
    let mut replay_buffer = PrioritizedReplayBuffer::new(
        config.replay_buffer_size,
        config.priority_alpha,
        config.priority_beta,
    );

    let mut agent = AgentFactory::create(
        config.algorithm,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    )?;

    // TODO: correctly implement VERIFY initialization
    // if !agent.online_network.verify_initialization()? {
    //     return Err(ExtractionError::ModelError(
    //         "Model initialization failed - weights are all zeros!".to_string()
    //     ));
    // }
    // info!("Model initialized successfully with non-zero weights");

    let mut env = ArticleExtractionEnvironment::new(baseline_extractor, config.clone());
    let mut metrics = TrainingMetrics::default();
    let mut epsilon = config.epsilon_start;

    // Initialize checkpoint manager
    let checkpoint_dir = config.models_dir.join("checkpoints");
    let checkpoint_manager = CheckpointManager::new(checkpoint_dir, 5)?;

    // CRITICAL FIX: Only resume if checkpoint is valid and from compatible run
    let start_episode = match checkpoint_manager.load_latest() {
        Ok(Some(checkpoint)) => {
            // VALIDATION: Check if checkpoint is compatible
            if checkpoint.episode >= config.num_episodes {
                warn!(
                    "Found checkpoint at episode {} but current run is only {} episodes. Starting fresh.",
                    checkpoint.episode, config.num_episodes
                );
                0
            } else if !checkpoint.model_path.exists() {
                warn!(
                    "Checkpoint references missing model file: {}. Starting fresh.",
                    checkpoint.model_path.display()
                );
                0
            } else {
                // Try to load the model
                info!("Found checkpoint at episode {}, attempting to load...", checkpoint.episode);

                match AgentFactory::load(
                    &checkpoint.model_path,
                    config.state_dim,
                    config.num_discrete_actions,
                    config.num_continuous_params,
                    &device,
                ) {
                    Ok(loaded_agent) => {
                        agent = loaded_agent;
                        epsilon = checkpoint.epsilon as f64;
                        metrics.best_avg_quality = checkpoint.best_quality;
                        info!("Successfully resumed from checkpoint at episode {}", checkpoint.episode);
                        checkpoint.episode
                    }
                    Err(e) => {
                        warn!("Failed to load checkpoint model: {}. Starting fresh.", e);
                        warn!("Consider deleting checkpoint directory if corruption persists.");
                        0
                    }
                }
            }
        }
        Ok(None) => {
            info!("No checkpoint found, starting fresh training");
            0
        }
        Err(e) => {
            warn!("Error loading checkpoint: {}. Starting fresh.", e);
            0
        }
    };

    // Progress bar
    let pb = ProgressBar::new((config.num_episodes - start_episode) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    for episode in start_episode..config.num_episodes {
        // Sample HTML
        let idx = episode % html_samples.len();
        let (html, url) = &html_samples[idx];

        let domain = extract_domain_from_url(url);

        let site_profile = site_memory.get_profile(&domain);

        // Reset environment
        let mut state = env.reset(html, url.clone(), Some(site_profile))?;

        let mut episode_reward = 0.0;
        let mut done = false;
        let mut step_info = StepInfo {
            quality_score: 0.0,
            text: String::new(),
            xpath: String::new(),
            parameters: std::collections::HashMap::new(),
            step_count: 0,
        };

        // Episode loop
        while !done {
            let action = agent.select_action(&state, epsilon as f32)?;
            let (next_state, reward, is_done, info) = env.step(action.clone())?;

            episode_reward += reward;
            done = is_done;
            step_info = info;

            // Store experience
            let experience = crate::replay_buffer::Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            };
            replay_buffer.add(experience);

            // Training step
            if replay_buffer.len() > config.batch_size * 10 {
                let loss = agent.train_step(&mut replay_buffer, config.batch_size)?;
                metrics.episode_losses.push(loss);
            }

            state = next_state;
        }

        // Update site profile
        let profile = site_memory.get_profile(&domain);
        let extraction_result = crate::site_profile::ExtractionResult {
            text: step_info.text.clone(),
            xpath: step_info.xpath.clone(),
            quality_score: step_info.quality_score,
            parameters: step_info.parameters.clone(),
            title: None,
            date: None,
        };
        profile.add_extraction(extraction_result);

        // Decay epsilon
        epsilon *= config.epsilon_decay;
        epsilon = epsilon.max(config.epsilon_end);

        // Update target network
        if episode % config.target_update_freq == 0 {
            agent.update_target_network();
        }

        // Record metrics
        metrics.episode_rewards.push(episode_reward);
        metrics.episode_qualities.push(step_info.quality_score);

        // Update progress bar
        if episode % 10 == 0 {
            let avg_reward = if metrics.episode_rewards.len() >= 100 {
                metrics.episode_rewards[metrics.episode_rewards.len() - 100..]
                    .iter()
                    .sum::<f32>() / 100.0
            } else {
                episode_reward
            };

            pb.set_message(format!(
                "Reward: {:.3}, Quality: {:.3}",
                avg_reward, step_info.quality_score
            ));
        }
        pb.inc(1);

        // Save checkpoint every checkpoint_freq episodes (only for long runs)
        if episode % config.checkpoint_freq == 0 && episode > 0 && config.num_episodes >= 5000 {
            let checkpoint_path = config.models_dir.join(format!(
                "checkpoint_{}_{}_ep{}.onnx",
                config.algorithm.to_string().to_lowercase(),
                chrono::Utc::now().format("%Y%m%d_%H%M%S"),
                episode
            ));

            // Validate save was successful
            match agent.save(&checkpoint_path) {
                Ok(_) => {
                    let avg_reward = if metrics.episode_rewards.len() >= 100 {
                        metrics.episode_rewards[metrics.episode_rewards.len() - 100..]
                            .iter()
                            .sum::<f32>() / 100.0
                    } else {
                        0.0
                    };

                    let avg_quality = if metrics.episode_qualities.len() >= 100 {
                        metrics.episode_qualities[metrics.episode_qualities.len() - 100..]
                            .iter()
                            .sum::<f32>() / 100.0
                    } else {
                        0.0
                    };

                    let checkpoint = Checkpoint::new(
                        episode,
                        agent.get_step_count(),
                        avg_reward,
                        avg_quality,
                        metrics.best_avg_quality,
                        epsilon as f32,
                        checkpoint_path.clone(),
                    );

                    match checkpoint_manager.save_checkpoint(&checkpoint) {
                        Ok(_) => {
                            if checkpoint_path.exists() {
                                let metadata = std::fs::metadata(&checkpoint_path)?;
                                if metadata.len() > 0 {
                                    site_memory.save_all()?;
                                    info!("Checkpoint saved at episode {} ({} bytes)", episode, metadata.len());
                                } else {
                                    warn!("Checkpoint file is empty, may be corrupted");
                                }
                            } else {
                                warn!("Checkpoint file disappeared after save");
                            }
                        }
                        Err(e) => {
                            warn!("Failed to save checkpoint metadata: {}", e);
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to save model checkpoint: {}", e);
                }
            }
        }
    }

    pb.finish_with_message("Training completed");

    // Save final model with validation, metadata and with algorithm name
    let final_path = config.models_dir.join(format!(
        "final_model_{}.onnx",
        config.algorithm.to_string().to_lowercase()
    ));

    let mut hyperparams = std::collections::HashMap::new();
    hyperparams.insert("learning_rate".to_string(), config.learning_rate);
    hyperparams.insert("batch_size".to_string(), config.batch_size as f64);
    hyperparams.insert("gamma".to_string(), config.gamma);
    hyperparams.insert("epsilon_decay".to_string(), config.epsilon_decay);
    hyperparams.insert("target_update_freq".to_string(), config.target_update_freq as f64);
    agent.save_with_metadata(&final_path, config.num_episodes, hyperparams)?;

    // Verify final save
    if final_path.exists() {
        let metadata = std::fs::metadata(&final_path)?;
        info!("Final model saved: {} bytes", metadata.len());
    }

    // Display metadata
    if let Ok(model_meta) = crate::models::ModelMetadata::load_metadata(&final_path) {
        model_meta.display();
    }

    site_memory.save_all()?;

    // Save final checkpoint with algorithm-specific path
    let final_checkpoint = Checkpoint::new(
        config.num_episodes,
        agent.get_step_count(),
        metrics.episode_rewards.last().copied().unwrap_or(0.0),
        metrics.episode_qualities.last().copied().unwrap_or(0.0),
        metrics.best_avg_quality,
        epsilon as f32,
        final_path,
    );
    checkpoint_manager.save_checkpoint(&final_checkpoint)?;

    info!("Training completed. Best avg quality: {:.3}", metrics.best_avg_quality);

    Ok((agent, metrics))
}


/// Training with improvements (curriculum learning, improved rewards, domain extraction, etc.)
pub fn train_with_improvements(
    config: &Config,
    html_samples: Vec<(String, String)>,
) -> Result<(Box<dyn RLAgent>, TrainingMetrics)> {
    info!("Starting OPTIMIZED training for {} episodes", config.num_episodes);
    info!("Performance settings:");
    info!("  - Batch size: {}", config.batch_size);
    info!("  - Train frequency: every {} steps", config.train_freq);
    info!("  - Gradient updates per episode: {}", config.num_train_steps_per_episode);
    info!("  - Min replay size: {}", config.min_replay_size);
    info!("  - Metrics window: {}", config.metrics_window);
    info!("  - Dataset size: {}", html_samples.len());

    // Initialize device and varbuilder
    let device = if config.use_cpu_for_tuning {
        Device::Cpu  // Force CPU for hyperparameter tuning
    } else if crate::cuda_is_available() {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };

    // step counters:
    let mut global_step:usize = 0;
    let mut total_training_steps:usize = 0;

    // Initialize components
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());
    let mut site_memory = SiteProfileMemory::new(&config.site_profiles_dir)?;
    let mut replay_buffer = PrioritizedReplayBuffer::new(
        config.replay_buffer_size,
        config.priority_alpha,
        config.priority_beta,
    );

    // varmap is created internally by AgentFactory
    let mut agent = AgentFactory::create(
        config.algorithm,
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.gamma as f32,
        config.learning_rate,
        &device,
    )?;

    let mut env = ArticleExtractionEnvironment::new(baseline_extractor.clone(), config.clone());
    let mut metrics = TrainingMetrics { episode_rewards: vec![], episode_qualities: vec![], episode_losses: vec![], best_avg_quality: 0.0 };

    // Enhanced components
    let reward_calculator = ImprovedRewardCalculator::new(config.stopwords.clone());
    let mut curriculum = CurriculumManager::new();
    let mut epsilon = config.epsilon_start;

    // ADDED: Checkpoint manager for improved training
    let checkpoint_dir = config.models_dir.join("checkpoints");
    let checkpoint_manager = CheckpointManager::new(checkpoint_dir, 5)?;

    // ADDED: Resume logic similar to train_standard
    let start_episode = match checkpoint_manager.load_latest() {
        Ok(Some(checkpoint)) => {
            // Validate checkpoint compatibility
            if checkpoint.episode >= config.num_episodes {
                warn!(
                    "Found checkpoint at episode {} but current run is only {} episodes. Starting fresh.",
                    checkpoint.episode, config.num_episodes
                );
                0
            } else if !checkpoint.model_path.exists() {
                warn!(
                    "Checkpoint references missing model file: {}. Starting fresh.",
                    checkpoint.model_path.display()
                );
                0
            } else {
                // Try to load the model
                info!("Found checkpoint at episode {}, attempting to load...", checkpoint.episode);

                match AgentFactory::load(
                    &checkpoint.model_path,
                    config.state_dim,
                    config.num_discrete_actions,
                    config.num_continuous_params,
                    &device,
                ) {
                    Ok(loaded_agent) => {
                        agent = loaded_agent;
                        epsilon = checkpoint.epsilon as f64;
                        metrics.best_avg_quality = checkpoint.best_quality;

                        // Try to load step counts from a separate file
                        let step_counts_path = checkpoint.model_path.with_extension("steps.json");
                        if step_counts_path.exists() {
                            if let Ok(step_data) = std::fs::read_to_string(&step_counts_path) {
                                if let Ok(step_counts) = serde_json::from_str::<(usize, usize)>(&step_data) {
                                    global_step = step_counts.0;
                                    total_training_steps = step_counts.1;
                                    info!("Resumed step counts: global_step={}, total_training_steps={}",
                                          global_step, total_training_steps);
                                }
                            }
                        }

                        info!("Successfully resumed from checkpoint at episode {}", checkpoint.episode);
                        checkpoint.episode
                    }
                    Err(e) => {
                        warn!("Failed to load checkpoint model: {}. Starting fresh.", e);
                        warn!("Consider deleting checkpoint directory if corruption persists.");
                        0
                    }
                }
            }
        }
        Ok(None) => {
            info!("No checkpoint found, starting fresh training");
            0
        }
        Err(e) => {
            warn!("Error loading checkpoint: {}. Starting fresh.", e);
            0
        }
    };

    // Progress bar - start from resume point
    let pb = ProgressBar::new((config.num_episodes - start_episode) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("█▓▒░"),
    );

    for episode in start_episode..config.num_episodes {
        let mut _episode_training_steps:usize = 0;
        // Update curriculum
        curriculum.update_threshold(episode);

        // Sample HTML (with curriculum filtering)
        let appropriate_samples: Vec<_> = html_samples.iter()
            .filter(|(html, _)| curriculum.is_appropriate(html))
            .collect();

        if appropriate_samples.is_empty() {
            warn!("No appropriate HTML samples for current curriculum");
            break;
        }

        let idx = episode % appropriate_samples.len();
        let (html, file_path) = appropriate_samples[idx];

        // Extract domain from ground truth JSON
        let domain = extract_domain_from_url(file_path);

        // Log domain extraction (first few episodes for verification)
        if episode < 10 {
            info!("Episode {}: File: {}, Domain: {}", episode, file_path, domain);
        }

        let site_profile = site_memory.get_profile(&domain);

        // Get baseline score
        let baseline_result = baseline_extractor.extract(html)?;
        let baseline_score = baseline_result.quality_score;

        // Reset environment
        let mut state = env.reset(html, file_path.clone(), Some(site_profile))?;

        let mut episode_reward = 0.0;
        let mut done = false;
        let mut step_info = StepInfo {
            quality_score: 0.0,
            text: String::new(),
            xpath: String::new(),
            parameters: std::collections::HashMap::new(),
            step_count: 0,
        };

        // Episode loop
        while !done {
            let action = agent.select_action(&state, epsilon as f32)?;
            let (next_state, _, is_done, info) = env.step(action.clone())?;

            // Calculate improved reward
            let reward = reward_calculator.calculate_reward(&info.text, baseline_score);

            episode_reward += reward;
            done = is_done;
            step_info = info;
            // Increment global step counter
            global_step += 1;

            // Store experience
            let experience = crate::replay_buffer::Experience {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            };
            replay_buffer.add(experience);

            // OPTIMIZED: More frequent training after warmup
            if replay_buffer.len() >= config.min_replay_size &&
                global_step % config.train_freq == 0 {
                // ADDED: Robust error handling for training step
                match agent.train_step(&mut replay_buffer, config.batch_size) {
                    Ok(loss) => {
                        // Check for NaN or infinite loss
                        if loss.is_nan() || loss.is_infinite() {
                            warn!("Invalid loss detected at episode {}, step {}: {}", episode, global_step, loss);
                            warn!("Skipping this training step");
                        } else {
                            metrics.episode_losses.push(loss);
                            _episode_training_steps += 1;
                        }
                    }
                    Err(e) => {
                        warn!("Training step failed at episode {}, step {}: {}", episode, global_step, e);
                        warn!("Continuing training...");
                        // Don't fail the entire run for one bad batch
                    }
                }
            }

            state = next_state;
        }

        // OPTIMIZED: Multiple gradient updates per episode
        if replay_buffer.len() >= config.min_replay_size {
            for update_idx in 0..config.num_train_steps_per_episode {
                match agent.train_step(&mut replay_buffer, config.batch_size) {
                    Ok(loss) => {
                        if loss.is_nan() || loss.is_infinite() {
                            warn!("Invalid loss in gradient update {} at episode {}", update_idx, episode);
                            break; // Stop further updates this episode
                        }
                        metrics.episode_losses.push(loss);
                        total_training_steps += 1;
                    }
                    Err(e) => {
                        warn!("Gradient update {} failed at episode {}: {}", update_idx, episode, e);
                        break; // Stop further updates this episode
                    }
                }
            }
        }

        // Update site profile with correct domain
        let profile = site_memory.get_profile(&domain);
        let extraction_result = crate::site_profile::ExtractionResult {
            text: step_info.text.clone(),
            xpath: step_info.xpath.clone(),
            quality_score: step_info.quality_score,
            parameters: step_info.parameters.clone(),
            title: None,
            date: None,
        };
        profile.add_extraction(extraction_result);
        // Save site profiles periodically
        if episode % 100 == 0 && episode > 0 {
            match site_memory.save_all() {
                Ok(_) => {
                    if episode % 500 == 0 {
                        info!("Site profiles saved at episode {}", episode);
                    }
                }
                Err(e) => {
                    warn!("Failed to save site profiles: {}", e);
                }
            }
        }

        // Decay epsilon (exponential)
        let progress = (episode as f32 / 2000.0).min(1.0);
        epsilon = config.epsilon_start * (config.epsilon_end / config.epsilon_start).powf(progress as f64);
        epsilon = epsilon.max(config.epsilon_end);

        // Update target network
        if episode % config.target_update_freq == 0 {
            agent.update_target_network();
        }

        // Record metrics
        metrics.episode_rewards.push(episode_reward);
        metrics.episode_qualities.push(step_info.quality_score);

        // Update progress bar
        if episode % config.log_freq == 0 {
            let window = config.metrics_window;
            let avg_reward = if metrics.episode_rewards.len() >= window {
                metrics.episode_rewards[metrics.episode_rewards.len() - window..]
                    .iter()
                    .sum::<f32>() / window as f32
            } else if !metrics.episode_rewards.is_empty() {
                metrics.episode_rewards.iter().sum::<f32>() / metrics.episode_rewards.len() as f32
            } else {
                0.0
            };

            let avg_quality = if metrics.episode_qualities.len() >= window {
                metrics.episode_qualities[metrics.episode_qualities.len() - window..]
                    .iter()
                    .sum::<f32>() / window as f32
            } else if !metrics.episode_qualities.is_empty() {
                metrics.episode_qualities.iter().sum::<f32>() / metrics.episode_qualities.len() as f32
            } else {
                0.0
            };

            let curriculum_threshold = curriculum.get_threshold();
            pb.set_message(format!(
                "R:{:.2} Q:{:.3} ε:{:.3} C:{:.2} Steps:{}",
                avg_reward, avg_quality, epsilon, curriculum_threshold, total_training_steps
            ));
        }
        pb.inc(1);

        // Save checkpoint every 500 episodes (more frequent for safety)
        if episode % config.checkpoint_freq == 0 && episode > 0 {
            let checkpoint_path = config.models_dir.join(format!(
                "checkpoint_{}_{}_ep{}.onnx",
                config.algorithm.to_string().to_lowercase(),
                chrono::Utc::now().format("%Y%m%d_%H%M%S"),
                episode
            ));

            match agent.save(&checkpoint_path) {
                Ok(_) => {
                    // Save step counts alongside model
                    let step_counts_path = checkpoint_path.with_extension("steps.json");
                    let step_counts = (global_step, total_training_steps);
                    if let Ok(step_data) = serde_json::to_string(&step_counts) {
                        let _ = std::fs::write(&step_counts_path, step_data);
                    }

                    let avg_reward = if metrics.episode_rewards.len() >= 100 {
                        metrics.episode_rewards[metrics.episode_rewards.len() - 100..]
                            .iter()
                            .sum::<f32>() / 100.0
                    } else {
                        0.0
                    };

                    let avg_quality = if metrics.episode_qualities.len() >= 100 {
                        metrics.episode_qualities[metrics.episode_qualities.len() - 100..]
                            .iter()
                            .sum::<f32>() / 100.0
                    } else {
                        0.0
                    };

                    let checkpoint = Checkpoint::new(
                        episode,
                        total_training_steps,
                        avg_reward,
                        avg_quality,
                        metrics.best_avg_quality,
                        epsilon as f32,
                        checkpoint_path.clone(),
                    );

                    match checkpoint_manager.save_checkpoint(&checkpoint) {
                        Ok(_) => {
                            site_memory.save_all()?;
                            info!("Improved training checkpoint saved at episode {} (global_step={})",
                                  episode, global_step);
                        }
                        Err(e) => {
                            warn!("Failed to save checkpoint metadata: {}", e);
                        }
                    }

                    if let Ok(metadata) = std::fs::metadata(&checkpoint_path) {
                        let file_size = metadata.len();
                        if file_size < 10_000 {
                            warn!("Checkpoint file suspiciously small: {} bytes", file_size);
                        } else {
                            info!("Checkpoint saved at episode {} ({} bytes)", episode, file_size);
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to save model checkpoint: {}", e);
                }
            }
        }

        // Track best model with algorithm-specific name
        if metrics.episode_qualities.len() >= 100 {
            let avg_quality = metrics.episode_qualities[metrics.episode_qualities.len() - 100..]
                .iter()
                .sum::<f32>() / 100.0;

            if avg_quality > metrics.best_avg_quality {
                metrics.best_avg_quality = avg_quality;
                let best_path = config.models_dir.join(format!(
                    "best_model_{}.onnx",
                    config.algorithm.to_string().to_lowercase()
                ));

                match agent.save(&best_path) {
                    Ok(_) => {
                        if let Ok(metadata) = std::fs::metadata(&best_path) {
                            info!("New best {} model saved with quality: {:.3} ({} bytes)",
                                  config.algorithm, avg_quality, metadata.len());
                        } else {
                            info!("New best {} model saved with quality: {:.3}",
                                  config.algorithm, avg_quality);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to save best model: {}", e);
                    }
                }
            }
        }
    }

    pb.finish_with_message("Improved training completed");

    // Save final model with validation, metadata and algorithm name
    let final_path = config.models_dir.join(format!(
        "final_model_{}.onnx",
        config.algorithm.to_string().to_lowercase()
    ));

    let mut hyperparams = std::collections::HashMap::new();
    hyperparams.insert("learning_rate".to_string(), config.learning_rate);
    hyperparams.insert("batch_size".to_string(), config.batch_size as f64);
    hyperparams.insert("gamma".to_string(), config.gamma);
    hyperparams.insert("epsilon_decay".to_string(), config.epsilon_decay);
    hyperparams.insert("target_update_freq".to_string(), config.target_update_freq as f64);
    agent.save_with_metadata(&final_path, config.num_episodes, hyperparams)?;

    // Verify final save
    if final_path.exists() {
        let metadata = std::fs::metadata(&final_path)?;
        info!("Final model saved: {} bytes", metadata.len());
    }

    // Display metadata
    if let Ok(model_meta) = crate::models::ModelMetadata::load_metadata(&final_path) {
        model_meta.display();
    }

    site_memory.save_all()?;

    // Save final checkpoint
    let final_checkpoint = Checkpoint::new(
        config.num_episodes,
        total_training_steps,
        metrics.episode_rewards.last().copied().unwrap_or(0.0),
        metrics.episode_qualities.last().copied().unwrap_or(0.0),
        metrics.best_avg_quality,
        epsilon as f32,
        final_path,
    );
    checkpoint_manager.save_checkpoint(&final_checkpoint)?;

    info!("Training completed:");
    info!("  - Total episodes: {}", config.num_episodes);
    info!("  - Total training steps: {}", total_training_steps);
    info!("  - Best avg quality: {:.3}", metrics.best_avg_quality);
    info!("  - Final epsilon: {:.3}", epsilon);

    Ok((agent, metrics))
}

/// Save training plot
pub fn save_training_plot(metrics: &TrainingMetrics, output_path: &Path) -> Result<()> {
    use plotters::prelude::*;

    let root = BitMapBackend::new(output_path, (1200, 800))
        .into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| ExtractionError::ModelError(format!("Plot error: {}", e)))?;

    let areas = root.split_evenly((2, 1));
    let upper = &areas[0];
    let lower = &areas[1];

    // Plot rewards
    let max_episodes = metrics.episode_rewards.len();
    let max_reward = metrics.episode_rewards.iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_reward = metrics.episode_rewards.iter()
        .copied()
        .fold(f32::INFINITY, f32::min);

    let mut chart = ChartBuilder::on(upper)
        .caption("Episode Rewards", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_episodes, min_reward..max_reward)
        .map_err(|e| ExtractionError::ModelError(format!("Chart error: {}", e)))?;

    chart.configure_mesh()
        .draw()
        .map_err(|e| ExtractionError::ModelError(format!("Mesh error: {}", e)))?;

    chart.draw_series(LineSeries::new(
        metrics.episode_rewards.iter().enumerate().map(|(i, &r)| (i, r)),
        &BLUE,
    ))
        .map_err(|e| ExtractionError::ModelError(format!("Series error: {}", e)))?;

    // Plot qualities
    let max_quality = metrics.episode_qualities.iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut chart2 = ChartBuilder::on(lower)
        .caption("Episode Quality", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_episodes, 0.0..max_quality)
        .map_err(|e| ExtractionError::ModelError(format!("Chart error: {}", e)))?;

    chart2.configure_mesh()
        .draw()
        .map_err(|e| ExtractionError::ModelError(format!("Mesh error: {}", e)))?;

    chart2.draw_series(LineSeries::new(
        metrics.episode_qualities.iter().enumerate().map(|(i, &q)| (i, q)),
        &GREEN,
    ))
        .map_err(|e| ExtractionError::ModelError(format!("Series error: {}", e)))?;

    root.present().map_err(|e| crate::ExtractionError::IoError(
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    ))?;

    info!("Training plot saved to: {}", output_path.display());

    Ok(())
}