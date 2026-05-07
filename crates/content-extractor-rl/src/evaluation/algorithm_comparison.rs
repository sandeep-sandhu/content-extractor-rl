// ============================================================================
// FILE: crates/content-extractor-rl/src/evaluation/algorithm_comparison.rs
// ============================================================================

use crate::{
    Config, Result, agents::AlgorithmType,
    training::{train_standard},
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::info;

/// Results for a single algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmResult {
    pub algorithm: AlgorithmType,
    pub runs: Vec<RunResult>,
    pub avg_quality: f64,
    pub std_quality: f64,
    pub avg_reward: f64,
    pub std_reward: f64,
    pub avg_training_time: f64,
}

/// Results for a single run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub run_number: usize,
    pub final_quality: f32,
    pub final_reward: f32,
    pub avg_quality_last_100: f32,
    pub avg_reward_last_100: f32,
    pub training_time_seconds: f64,
}

/// Comparison report across all algorithms
#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub algorithms: Vec<AlgorithmResult>,
    pub best_by_quality: String,
    pub best_by_reward: String,
    pub best_by_time: String,
    pub config: ComparisonConfig,
}

/// Configuration for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    pub episodes: usize,
    pub runs: usize,
    pub dataset_size: usize,
}

/// Algorithm comparator
pub struct AlgorithmComparator {
    config: Config,
    output_dir: PathBuf,
}

impl AlgorithmComparator {
    pub fn new(config: Config, output_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&output_dir)?;
        Ok(Self { config, output_dir })
    }

    /// Compare multiple algorithms
    pub fn compare_algorithms(
        &self,
        algorithms: Vec<AlgorithmType>,
        html_samples: Vec<(String, String)>,
        episodes: usize,
        runs: usize,
    ) -> Result<ComparisonReport> {
        info!("Starting algorithm comparison");
        info!("Algorithms: {:?}", algorithms);
        info!("Episodes: {}, Runs per algorithm: {}", episodes, runs);

        let mut results = Vec::new();

        for algorithm in algorithms {
            info!("Evaluating algorithm: {}", algorithm);
            let algo_result = self.evaluate_algorithm(
                algorithm,
                html_samples.clone(),
                episodes,
                runs,
            )?;
            results.push(algo_result);
        }

        // Find best algorithms
        let best_by_quality = results.iter()
            .max_by(|a, b| a.avg_quality.partial_cmp(&b.avg_quality).unwrap())
            .map(|r| r.algorithm.to_string())
            .unwrap_or_else(|| "None".to_string());

        let best_by_reward = results.iter()
            .max_by(|a, b| a.avg_reward.partial_cmp(&b.avg_reward).unwrap())
            .map(|r| r.algorithm.to_string())
            .unwrap_or_else(|| "None".to_string());

        let best_by_time = results.iter()
            .min_by(|a, b| a.avg_training_time.partial_cmp(&b.avg_training_time).unwrap())
            .map(|r| r.algorithm.to_string())
            .unwrap_or_else(|| "None".to_string());

        let report = ComparisonReport {
            algorithms: results,
            best_by_quality: best_by_quality.clone(),
            best_by_reward: best_by_reward.clone(),
            best_by_time: best_by_time.clone(),
            config: ComparisonConfig {
                episodes,
                runs,
                dataset_size: html_samples.len(),
            },
        };

        // Save report
        self.save_report(&report)?;

        // Print summary
        self.print_summary(&report);

        Ok(report)
    }

    /// Evaluate a single algorithm with multiple runs
    fn evaluate_algorithm(
        &self,
        algorithm: AlgorithmType,
        html_samples: Vec<(String, String)>,
        episodes: usize,
        runs: usize,
    ) -> Result<AlgorithmResult> {
        let mut run_results = Vec::new();

        for run in 0..runs {
            info!("Algorithm: {}, Run: {}/{}", algorithm, run + 1, runs);

            let start_time = std::time::Instant::now();

            // Create config for this run
            let mut run_config = self.config.clone();
            run_config.algorithm = algorithm;
            run_config.num_episodes = episodes;

            // Train
            let (_agent, metrics) = train_standard(&run_config, html_samples.clone())?;

            let training_time = start_time.elapsed().as_secs_f64();

            // Calculate metrics
            let final_quality = metrics.episode_qualities.last().copied().unwrap_or(0.0);
            let final_reward = metrics.episode_rewards.last().copied().unwrap_or(0.0);

            let avg_quality_last_100 = if metrics.episode_qualities.len() >= 100 {
                metrics.episode_qualities[metrics.episode_qualities.len() - 100..]
                    .iter()
                    .sum::<f32>() / 100.0
            } else if !metrics.episode_qualities.is_empty() {
                metrics.episode_qualities.iter().sum::<f32>() / metrics.episode_qualities.len() as f32
            } else {
                0.0
            };

            let avg_reward_last_100 = if metrics.episode_rewards.len() >= 100 {
                metrics.episode_rewards[metrics.episode_rewards.len() - 100..]
                    .iter()
                    .sum::<f32>() / 100.0
            } else if !metrics.episode_rewards.is_empty() {
                metrics.episode_rewards.iter().sum::<f32>() / metrics.episode_rewards.len() as f32
            } else {
                0.0
            };

            let run_result = RunResult {
                run_number: run,
                final_quality,
                final_reward,
                avg_quality_last_100,
                avg_reward_last_100,
                training_time_seconds: training_time,
            };

            run_results.push(run_result);

            info!("Run {} complete: quality={:.4}, reward={:.4}, time={:.2}s",
                  run + 1, avg_quality_last_100, avg_reward_last_100, training_time);
        }

        // Calculate statistics
        let avg_quality = run_results.iter()
            .map(|r| r.avg_quality_last_100 as f64)
            .sum::<f64>() / runs as f64;

        let std_quality = {
            let variance = run_results.iter()
                .map(|r| {
                    let diff = r.avg_quality_last_100 as f64 - avg_quality;
                    diff * diff
                })
                .sum::<f64>() / runs as f64;
            variance.sqrt()
        };

        let avg_reward = run_results.iter()
            .map(|r| r.avg_reward_last_100 as f64)
            .sum::<f64>() / runs as f64;

        let std_reward = {
            let variance = run_results.iter()
                .map(|r| {
                    let diff = r.avg_reward_last_100 as f64 - avg_reward;
                    diff * diff
                })
                .sum::<f64>() / runs as f64;
            variance.sqrt()
        };

        let avg_training_time = run_results.iter()
            .map(|r| r.training_time_seconds)
            .sum::<f64>() / runs as f64;

        Ok(AlgorithmResult {
            algorithm,
            runs: run_results,
            avg_quality,
            std_quality,
            avg_reward,
            std_reward,
            avg_training_time,
        })
    }

    /// Save comparison report
    fn save_report(&self, report: &ComparisonReport) -> Result<()> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let path = self.output_dir.join(format!("comparison_report_{}.json", timestamp));

        let json = serde_json::to_string_pretty(report)?;
        std::fs::write(&path, json)?;

        info!("Comparison report saved to: {}", path.display());
        Ok(())
    }

    /// Print summary to console
    fn print_summary(&self, report: &ComparisonReport) {
        println!("\n{}", "=".repeat(80));
        println!("ALGORITHM COMPARISON RESULTS");
        println!("{}", "=".repeat(80));
        println!("Episodes: {}, Runs per algorithm: {}",
                 report.config.episodes, report.config.runs);
        println!("Dataset size: {}", report.config.dataset_size);
        println!("{}", "=".repeat(80));

        for result in &report.algorithms {
            println!("\nAlgorithm: {}", result.algorithm);
            println!("  Average Quality:  {:.4} ± {:.4}", result.avg_quality, result.std_quality);
            println!("  Average Reward:   {:.4} ± {:.4}", result.avg_reward, result.std_reward);
            println!("  Average Time:     {:.2}s", result.avg_training_time);
            println!("  Individual runs:");
            for run in &result.runs {
                println!("    Run {}: quality={:.4}, reward={:.4}, time={:.2}s",
                         run.run_number + 1,
                         run.avg_quality_last_100,
                         run.avg_reward_last_100,
                         run.training_time_seconds);
            }
        }

        println!("\n{}", "=".repeat(80));
        println!("WINNERS");
        println!("{}", "=".repeat(80));
        println!("Best Quality:  {}", report.best_by_quality);
        println!("Best Reward:   {}", report.best_by_reward);
        println!("Fastest:       {}", report.best_by_time);
        println!("{}", "=".repeat(80));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_comparison_report_creation() {
        let run_result = RunResult {
            run_number: 0,
            final_quality: 0.8,
            final_reward: 10.0,
            avg_quality_last_100: 0.75,
            avg_reward_last_100: 9.5,
            training_time_seconds: 100.0,
        };

        let algo_result = AlgorithmResult {
            algorithm: AlgorithmType::DuelingDQN,
            runs: vec![run_result],
            avg_quality: 0.75,
            std_quality: 0.05,
            avg_reward: 9.5,
            std_reward: 0.5,
            avg_training_time: 100.0,
        };

        assert_eq!(algo_result.runs.len(), 1);
        assert!((algo_result.avg_quality - 0.75).abs() < 0.01);
    }
}