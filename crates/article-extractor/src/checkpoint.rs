//! Model checkpoint management

use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::fs;

/// Model checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub episode: usize,
    pub step_count: usize,
    pub avg_reward: f32,
    pub avg_quality: f32,
    pub best_quality: f32,
    pub epsilon: f32,
    pub timestamp: String,
    pub model_path: PathBuf,
    pub optimizer_state: Option<PathBuf>,
}

impl Checkpoint {
    /// Create new checkpoint
    pub fn new(
        episode: usize,
        step_count: usize,
        avg_reward: f32,
        avg_quality: f32,
        best_quality: f32,
        epsilon: f32,
        model_path: PathBuf,
    ) -> Self {
        Self {
            episode,
            step_count,
            avg_reward,
            avg_quality,
            best_quality,
            epsilon,
            timestamp: chrono::Utc::now().to_rfc3339(),
            model_path,
            optimizer_state: None,
        }
    }

    /// Save checkpoint to JSON
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load checkpoint from JSON
    pub fn load(path: &Path) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let checkpoint = serde_json::from_str(&json)?;
        Ok(checkpoint)
    }
}

/// Checkpoint manager
pub struct CheckpointManager {
    checkpoints_dir: PathBuf,
    max_checkpoints: usize,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(checkpoints_dir: PathBuf, max_checkpoints: usize) -> Result<Self> {
        fs::create_dir_all(&checkpoints_dir)?;
        Ok(Self {
            checkpoints_dir,
            max_checkpoints,
        })
    }

    /// Save checkpoint
    pub fn save_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
        let checkpoint_file = self.checkpoints_dir.join(format!(
            "checkpoint_ep{}.json",
            checkpoint.episode
        ));

        checkpoint.save(&checkpoint_file)?;

        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(())
    }

    /// Load latest checkpoint
    pub fn load_latest(&self) -> Result<Option<Checkpoint>> {
        let mut checkpoints = self.list_checkpoints()?;

        if checkpoints.is_empty() {
            return Ok(None);
        }

        checkpoints.sort_by_key(|c| c.episode);
        let latest = checkpoints.last().unwrap();

        Ok(Some(latest.clone()))
    }

    /// Load best checkpoint (by quality)
    pub fn load_best(&self) -> Result<Option<Checkpoint>> {
        let checkpoints = self.list_checkpoints()?;

        if checkpoints.is_empty() {
            return Ok(None);
        }

        let best = checkpoints.iter()
            .max_by(|a, b| a.best_quality.partial_cmp(&b.best_quality).unwrap())
            .cloned();

        Ok(best)
    }

    /// List all checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<Checkpoint>> {
        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(&self.checkpoints_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(checkpoint) = Checkpoint::load(&path) {
                    checkpoints.push(checkpoint);
                }
            }
        }

        Ok(checkpoints)
    }

    /// Clean up old checkpoints, keeping only the most recent ones
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let mut checkpoints = self.list_checkpoints()?;

        if checkpoints.len() <= self.max_checkpoints {
            return Ok(());
        }

        // Sort by episode
        checkpoints.sort_by_key(|c| c.episode);

        // Remove oldest checkpoints
        let to_remove = checkpoints.len() - self.max_checkpoints;

        for checkpoint in checkpoints.iter().take(to_remove) {
            let checkpoint_file = self.checkpoints_dir.join(format!(
                "checkpoint_ep{}.json",
                checkpoint.episode
            ));

            if checkpoint_file.exists() {
                fs::remove_file(checkpoint_file)?;
            }

            // Also remove model file if it exists
            if checkpoint.model_path.exists() {
                fs::remove_file(&checkpoint.model_path)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_path = temp_dir.path().join("checkpoint.json");

        let checkpoint = Checkpoint::new(
            100,
            5000,
            0.5,
            0.7,
            0.8,
            0.1,
            PathBuf::from("model.onnx"),
        );

        checkpoint.save(&checkpoint_path).unwrap();
        let loaded = Checkpoint::load(&checkpoint_path).unwrap();

        assert_eq!(loaded.episode, 100);
        assert_eq!(loaded.step_count, 5000);
    }

    #[test]
    fn test_checkpoint_manager() {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path().to_path_buf(), 3).unwrap();

        // Save multiple checkpoints
        for i in 0..5 {
            let checkpoint = Checkpoint::new(
                i * 100,
                i * 1000,
                0.5,
                0.7,
                0.8,
                0.1,
                PathBuf::from(format!("model_{}.onnx", i)),
            );
            manager.save_checkpoint(&checkpoint).unwrap();
        }

        // Should only keep 3 most recent
        let checkpoints = manager.list_checkpoints().unwrap();
        assert!(checkpoints.len() <= 3);
    }
}
