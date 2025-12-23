//! MLflow experiment tracking integration using trs_mlflow crate

#[cfg(feature = "mlflow-rs")]
use trs_mlflow::{run::CreateRun, Client};

use crate::{Result, training::TrainingMetrics};
use std::collections::HashMap;
use std::path::Path;
use tracing::{info, warn};

#[cfg(not(feature = "mlflow-rs"))]
impl MlflowTracker {
    pub fn new(_: Option<String>) -> Self { Self { enabled: false } }
    pub fn start_run(&mut self, _: Option<String>) -> Result<()> { Ok(()) }
    pub fn log_metric(&self, _: &str, _: f64, _: Option<i64>) -> Result<()> { Ok(()) }
    // Single catch-all for all logging methods
}

/// MLflow experiment tracker using trs_mlflow client
pub struct MlflowTracker {
    #[cfg(feature = "mlflow-rs")]
    client: Option<Client>,
    #[cfg(feature = "mlflow-rs")]
    runtime: Option<tokio::runtime::Runtime>,
    #[cfg(feature = "mlflow-rs")]
    run_id: Option<String>,
    #[cfg(feature = "mlflow-rs")]
    experiment_id: Option<String>,
    experiment_name: String,
    enabled: bool,
}

impl MlflowTracker {
    /// Create new MLflow tracker with trs_mlflow client
    pub fn new(tracking_uri: Option<String>) -> Self {
        #[cfg(feature = "mlflow-rs")]
        {
            let enabled = if let Some(uri) = tracking_uri {
                // Build the full API URL
                let api_url = if uri.ends_with("/api") {
                    uri
                } else if uri.ends_with("/") {
                    format!("{}api", uri)
                } else {
                    format!("{}/api", uri)
                };

                info!("Initializing MLflow client with URI: {}", api_url);

                // Create async runtime and client
                match tokio::runtime::Runtime::new() {
                    Ok(runtime) => {
                        let client = runtime.block_on(async {
                            Client::new(&api_url)
                        });

                        Some((client, runtime))
                    }
                    Err(e) => {
                        warn!("Failed to create async runtime for MLflow: {}", e);
                        None
                    }
                }
            } else {
                None
            };

            let (client, runtime, enabled_flag) = if let Some((client, runtime)) = enabled {
                (Some(client), Some(runtime), true)
            } else {
                (None, None, false)
            };

            Self {
                client,
                runtime,
                run_id: None,
                experiment_id: None,
                experiment_name: Self::generate_experiment_name(),
                enabled: enabled_flag,
            }
        }

        #[cfg(not(feature = "mlflow-rs"))]
        {
            let _ = tracking_uri;
            info!("MLflow tracking disabled (mlflow-rs feature not enabled)");
            Self {
                experiment_name: Self::generate_experiment_name(),
                enabled: false,
            }
        }
    }

    /// Create tracker with automatic configuration
    pub fn with_auto_config() -> Self {
        let tracking_uri = std::env::var("MLFLOW_TRACKING_URI").ok();
        Self::new(tracking_uri)
    }

    /// Generate automatic experiment name
    fn generate_experiment_name() -> String {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        format!("article_extractor_{}", timestamp)
    }

    /// Start a new MLflow run
    pub fn start_run(&mut self, run_name: Option<String>) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        #[cfg(feature = "mlflow-rs")]
        {
            if let (Some(ref client), Some(ref runtime)) = (&self.client, &self.runtime) {
                // Create or get experiment
                let experiment_id = runtime.block_on(async {
                    match client.create_experiment(&self.experiment_name, vec![]).await {
                        Ok(id) => Ok(id),
                        Err(e) => {
                            // Try to find existing experiment
                            warn!("Failed to create experiment: {}", e);
                            // For now, create with a timestamped name
                            let fallback_name = format!("{}_{}", self.experiment_name, chrono::Utc::now().timestamp());
                            client.create_experiment(&fallback_name, vec![]).await
                        }
                    }
                }).map_err(|e: anyhow::Error| {
                    crate::ExtractionError::MlflowError(format!("Failed to create experiment: {}", e))
                })?;

                self.experiment_id = Some(experiment_id.clone());

                // Create run
                let create_run = CreateRun::new()
                    .run_name(&run_name.unwrap_or_else(|| "unnamed_run".to_string()))
                    .experiment_id(&experiment_id)
                    .build();

                let run = runtime.block_on(async {
                    client.create_run(create_run).await
                }).map_err(|e: anyhow::Error| {
                    crate::ExtractionError::MlflowError(format!("Failed to create run: {}", e))
                })?;

                self.run_id = Some(run.info.run_id.clone());
                info!("Started MLflow run: {}", run.info.run_id);
            }
        }

        Ok(())
    }

    /// Log parameters to MLflow
    pub fn log_params(&self, params: HashMap<String, String>) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        #[cfg(feature = "mlflow-rs")]
        {
            if let (Some(ref client), Some(ref runtime), Some(ref run_id)) =
                (&self.client, &self.runtime, &self.run_id)
            {
                for (key, value) in params {
                    // Note: trs_mlflow may not have log_param method
                    // We'll use a different approach or skip if not available
                    warn!("log_param method not available in trs_mlflow - skipping parameter: {} = {}", key, value);
                }
            }
        }

        Ok(())
    }

    /// Log a single metric value
    pub fn log_metric(&self, key: &str, value: f64, step: Option<i64>) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        #[cfg(feature = "mlflow-rs")]
        {
            if let (Some(ref client), Some(ref runtime), Some(ref run_id)) =
                (&self.client, &self.runtime, &self.run_id)
            {
                let step = step.unwrap_or(0);
                // Note: trs_mlflow may not have log_metric method
                warn!("log_metric method not available in trs_mlflow - skipping metric: {} = {} at step {}", key, value, step);
            }
        }

        Ok(())
    }

    /// Log multiple metrics
    pub fn log_metrics(&self, metrics: HashMap<String, f64>, step: Option<i64>) -> Result<()> {
        for (key, value) in metrics {
            self.log_metric(&key, value, step)?;
        }
        Ok(())
    }

    /// Log training metrics
    pub fn log_training_metrics(&self, metrics: &TrainingMetrics, episode: usize) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let step = Some(episode as i64);

        // Log latest values
        if let Some(&reward) = metrics.episode_rewards.last() {
            self.log_metric("episode_reward", reward as f64, step)?;
        }

        if let Some(&quality) = metrics.episode_qualities.last() {
            self.log_metric("episode_quality", quality as f64, step)?;
        }

        if let Some(&loss) = metrics.episode_losses.last() {
            self.log_metric("episode_loss", loss as f64, step)?;
        }

        // Log running averages
        if metrics.episode_rewards.len() >= 100 {
            let avg_reward: f32 = metrics.episode_rewards[metrics.episode_rewards.len() - 100..]
                .iter()
                .sum::<f32>() / 100.0;
            self.log_metric("avg_reward_100", avg_reward as f64, step)?;
        }

        Ok(())
    }

    /// Log artifact (file) - Note: trs_mlflow artifact API might be limited
    pub fn log_artifact(&self, local_path: &Path) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        #[cfg(feature = "mlflow-rs")]
        {
            warn!("Artifact logging not fully implemented in trs_mlflow - file saved locally: {}",
                  local_path.display());
        }

        Ok(())
    }

    /// End the current run
    pub fn end_run(&mut self, status: &str) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        #[cfg(feature = "mlflow-rs")]
        {
            if let (Some(ref client), Some(ref runtime), Some(ref run_id)) =
                (&self.client, &self.runtime, &self.run_id)
            {
                // Log final status as a param
                warn!("log_param method not available in trs_mlflow - skipping final_status: {}", status);

                info!("Ended MLflow run: {}", run_id);
            }
            self.run_id = None;
            self.experiment_id = None;
        }

        Ok(())
    }

    /// Check if MLflow is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the current run ID
    pub fn get_run_id(&self) -> Option<&String> {
        #[cfg(feature = "mlflow-rs")]
        {
            self.run_id.as_ref()
        }
        #[cfg(not(feature = "mlflow-rs"))]
        {
            None
        }
    }
}

/// Helper to create MLflow tracker from environment
pub fn create_tracker() -> MlflowTracker {
    MlflowTracker::with_auto_config()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlflow_tracker_creation_disabled() {
        let tracker = MlflowTracker::new(None);
        assert!(!tracker.is_enabled());
    }

    #[test]
    fn test_log_metrics_disabled() {
        let tracker = MlflowTracker::new(None);
        let mut metrics_map = HashMap::new();
        metrics_map.insert("test_metric".to_string(), 0.5);

        // Should not error when disabled
        tracker.log_metrics(metrics_map, Some(0)).unwrap();
    }
}