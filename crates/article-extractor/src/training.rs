use crate::{
    Config, DQNAgent, ArticleExtractionEnvironment, BaselineExtractor,
    PrioritizedReplayBuffer, SiteProfileMemory, ImprovedRewardCalculator,
    CurriculumManager, Result,
};
use crate::environment::StepInfo;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use tracing::{info, warn};
use crate::{Checkpoint, CheckpointManager};

/// Training metrics
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub episode_rewards: Vec<f32>,
    pub episode_qualities: Vec<f32>,
    pub episode_losses: Vec<f32>,
    pub best_avg_quality: f32,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            episode_rewards: Vec::new(),
            episode_qualities: Vec::new(),
            episode_losses: Vec::new(),
            best_avg_quality: 0.0,
        }
    }
}


/// Standard training loop with checkpoint support
pub fn train_standard(
    config: &Config,
    html_samples: Vec<(String, String)>,
) -> Result<(DQNAgent, TrainingMetrics)> {
    info!("Starting standard training for {} episodes", config.num_episodes);

    // Initialize components
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());
    let mut site_memory = SiteProfileMemory::new(&config.site_profiles_dir)?;
    let mut replay_buffer = PrioritizedReplayBuffer::new(
        config.replay_buffer_size,
        config.priority_alpha,
        config.priority_beta,
    );

    let mut agent = DQNAgent::new(
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.learning_rate,
        config.gamma,
    )?;

    let mut env = ArticleExtractionEnvironment::new(baseline_extractor, config.clone());
    let mut metrics = TrainingMetrics::new();
    let mut epsilon = config.epsilon_start;

    // Initialize checkpoint manager
    let checkpoint_dir = config.models_dir.join("checkpoints");
    let checkpoint_manager = CheckpointManager::new(checkpoint_dir, 5)?;

    // Try to resume from checkpoint
    let start_episode = if let Some(checkpoint) = checkpoint_manager.load_latest()? {
        info!("Resuming from checkpoint at episode {}", checkpoint.episode);
        epsilon = checkpoint.epsilon;
        metrics.best_avg_quality = checkpoint.best_quality;

        // Load model
        if checkpoint.model_path.exists() {
            agent = DQNAgent::load(
                &checkpoint.model_path,
                config.state_dim,
                config.num_discrete_actions,
                config.num_continuous_params,
            )?;
        }

        checkpoint.episode
    } else {
        0
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
        // [Previous training loop code...]

        // Save checkpoint every 1000 episodes
        if episode % 1000 == 0 && episode > 0 {
            let checkpoint_path = config.models_dir.join(format!("checkpoint_ep{}.onnx", episode));
            agent.save(&checkpoint_path)?;

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
                agent.step_count,
                avg_reward,
                avg_quality,
                metrics.best_avg_quality,
                epsilon,
                checkpoint_path,
            );

            checkpoint_manager.save_checkpoint(&checkpoint)?;
            site_memory.save_all()?;
            info!("Saved checkpoint at episode {}", episode);
        }

        // [Rest of training loop...]
    }

    pb.finish_with_message("Training completed");

    // Save final model
    let final_path = config.models_dir.join("final_model.onnx");
    agent.save(&final_path)?;
    site_memory.save_all()?;

    // Save final checkpoint
    let final_checkpoint = Checkpoint::new(
        config.num_episodes,
        agent.step_count,
        metrics.episode_rewards.last().copied().unwrap_or(0.0),
        metrics.episode_qualities.last().copied().unwrap_or(0.0),
        metrics.best_avg_quality,
        epsilon,
        final_path,
    );
    checkpoint_manager.save_checkpoint(&final_checkpoint)?;

    info!("Training completed. Best avg quality: {:.3}", metrics.best_avg_quality);

    Ok((agent, metrics))
}


/// Training with improvements (curriculum learning, improved rewards, etc.)
pub fn train_with_improvements(
    config: &Config,
    html_samples: Vec<(String, String)>,
) -> Result<(DQNAgent, TrainingMetrics)> {
    info!("Starting improved training for {} episodes", config.num_episodes);

    // Initialize components
    let baseline_extractor = BaselineExtractor::new(config.stopwords.clone());
    let mut site_memory = SiteProfileMemory::new(&config.site_profiles_dir)?;
    let mut replay_buffer = PrioritizedReplayBuffer::new(
        config.replay_buffer_size,
        config.priority_alpha,
        config.priority_beta,
    );

    let mut agent = DQNAgent::new(
        config.state_dim,
        config.num_discrete_actions,
        config.num_continuous_params,
        config.learning_rate,
        config.gamma,
    )?;

    let mut env = ArticleExtractionEnvironment::new(baseline_extractor.clone(), config.clone());
    let mut metrics = TrainingMetrics::new();

    // Enhanced components
    let reward_calculator = ImprovedRewardCalculator::new(config.stopwords.clone());
    let mut curriculum = CurriculumManager::new();
    let mut epsilon = config.epsilon_start;

    // Progress bar
    let pb = ProgressBar::new(config.num_episodes as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );

    for episode in 0..config.num_episodes {
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
        let (html, url) = appropriate_samples[idx];

        let domain = url::Url::parse(url)
            .ok()
            .and_then(|u| u.host_str().map(|h| h.to_string()))
            .unwrap_or_else(|| "unknown".to_string());

        let site_profile = site_memory.get_profile(&domain);

        // Get baseline score
        let baseline_result = baseline_extractor.extract(html)?;
        let baseline_score = baseline_result.quality_score;

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
            let action = agent.select_action(&state, epsilon)?;
            let (next_state, _, is_done, info) = env.step(action.clone())?;

            // Calculate improved reward
            let reward = reward_calculator.calculate_reward(&info.text, baseline_score);

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
        };
        profile.add_extraction(extraction_result);

        // Decay epsilon (exponential)
        let progress = (episode as f32 / 2000.0).min(1.0);
        epsilon = config.epsilon_start as f32 * (config.epsilon_end as f32 / config.epsilon_start as f32).powf(progress);
        epsilon = epsilon.max(config.epsilon_end as f32);

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

            let curriculum_threshold = curriculum.get_threshold();
            pb.set_message(format!(
                "Reward: {:.3}, Quality: {:.3}, Curriculum: {:.2}",
                avg_reward, step_info.quality_score, curriculum_threshold
            ));
        }
        pb.inc(1);

        // Save checkpoint
        if episode % 1000 == 0 && episode > 0 {
            let checkpoint_path = config.models_dir.join(format!("checkpoint_ep{}.onnx", episode));
            agent.save(&checkpoint_path)?;
            site_memory.save_all()?;
            info!("Saved checkpoint at episode {}", episode);
        }

        // Track best model
        if metrics.episode_qualities.len() >= 100 {
            let avg_quality = metrics.episode_qualities[metrics.episode_qualities.len() - 100..]
                .iter()
                .sum::<f32>() / 100.0;

            if avg_quality > metrics.best_avg_quality {
                metrics.best_avg_quality = avg_quality;
                let best_path = config.models_dir.join("best_model.onnx");
                agent.save(&best_path)?;
                info!("New best model saved with quality: {:.3}", avg_quality);
            }
        }
    }

    pb.finish_with_message("Improved training completed");

    // Final save
    let final_path = config.models_dir.join("final_model.onnx");
    agent.save(&final_path)?;
    site_memory.save_all()?;

    info!("Training completed. Best avg quality: {:.3}", metrics.best_avg_quality);

    Ok((agent, metrics))
}

/// Save training plot
pub fn save_training_plot(metrics: &TrainingMetrics, output_path: &Path) -> Result<()> {
    use plotters::prelude::*;

    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_evenly(2);

    // Plot rewards
    let max_episodes = metrics.episode_rewards.len();
    let max_reward = metrics.episode_rewards.iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_reward = metrics.episode_rewards.iter()
        .copied()
        .fold(f32::INFINITY, f32::min);

    let mut chart = ChartBuilder::on(&upper)
        .caption("Episode Rewards", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_episodes, min_reward..max_reward)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        metrics.episode_rewards.iter().enumerate().map(|(i, &r)| (i, r)),
        &BLUE,
    ))?;

    // Plot qualities
    let max_quality = metrics.episode_qualities.iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut chart2 = ChartBuilder::on(&lower)
        .caption("Episode Quality", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_episodes, 0.0..max_quality)?;

    chart2.configure_mesh().draw()?;

    chart2.draw_series(LineSeries::new(
        metrics.episode_qualities.iter().enumerate().map(|(i, &q)| (i, q)),
        &GREEN,
    ))?;

    root.present().map_err(|e| crate::ExtractionError::IoError(
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    ))?;

    info!("Training plot saved to: {}", output_path.display());

    Ok(())
}
