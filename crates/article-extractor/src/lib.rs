//! Article Extractor - RL-based HTML article extraction library
//!
//! This library provides functionality for extracting article content from HTML
//! using reinforcement learning with fallback to heuristic-based extraction.

// ============================================================================
// FILE: crates/article-extractor/src/lib.rs
// ============================================================================

pub mod config;
pub mod text_utils;
pub mod html_parser;
pub mod site_profile;
pub mod baseline_extractor;
pub mod evaluation;

pub use evaluation::{
    GroundTruthData, GroundTruthEvaluator, EvaluationMetrics,
    algorithm_comparison::{AlgorithmComparator, ComparisonReport},
};
pub mod models;
pub use models::ModelMetadata;

pub mod environment;
pub mod replay_buffer;
pub mod reward;
pub mod curriculum;

pub mod agents;
pub use agents::{AgentFactory, AlgorithmType, RLAgent};
pub use cli_utils::*;
pub use checkpoint::{Checkpoint, CheckpointManager};

pub mod training;
pub mod hyperparameter_tuner;
pub mod plotting;
pub mod device;

// Optional MLflow integration
#[cfg(feature = "mlflow-rs")]
pub mod mlflow;

// Re-exports
pub use config::Config;
pub use site_profile::{SiteProfile, SiteProfileMemory};
pub use baseline_extractor::BaselineExtractor;
pub use environment::ArticleExtractionEnvironment;
pub use training::{train_standard, train_with_improvements, TrainingMetrics};
pub use hyperparameter_tuner::{TPEOptimizer, Hyperparameters, HyperparameterSpace, TrialResult};

pub use plotting::{TrainingPlotter, PlotConfig};
pub use device::{get_device, cuda_is_available, get_device_info, print_device_info};

pub mod checkpoint;
pub mod cli_utils;



#[cfg(feature = "mlflow-rs")]
pub use mlflow::{MlflowTracker, create_tracker};

/// Result type for article extraction operations
pub type Result<T> = std::result::Result<T, ExtractionError>;

/// Errors that can occur during article extraction
#[derive(Debug)]
pub enum ExtractionError {
    IoError(std::io::Error),
    ParseError(String),
    NetworkError(String),
    ModelError(String),
    ExtractionFailed(String),
    CandleError(String),
    RuntimeError(String),
    MlflowError(String),
}

impl From<anyhow::Error> for ExtractionError {
    fn from(err: anyhow::Error) -> Self {
        ExtractionError::MlflowError(format!("{}", err))
    }
}

impl std::fmt::Display for ExtractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractionError::IoError(e) => write!(f, "IO error: {}", e),
            ExtractionError::ParseError(e) => write!(f, "Parse error: {}", e),
            ExtractionError::NetworkError(e) => write!(f, "Network error: {}", e),
            ExtractionError::ModelError(e) => write!(f, "Model error: {}", e),
            ExtractionError::ExtractionFailed(e) => write!(f, "Extraction failed: {}", e),
            ExtractionError::CandleError(e) => write!(f, "Candle error: {}", e),
            ExtractionError::RuntimeError(e) => write!(f, "Runtime error: {}", e),
            ExtractionError::MlflowError(e) => write!(f, "MLFlow error: {}", e),
        }
    }
}

impl std::error::Error for ExtractionError {}

impl From<std::io::Error> for ExtractionError {
    fn from(err: std::io::Error) -> Self {
        ExtractionError::IoError(err)
    }
}

impl From<serde_json::Error> for ExtractionError {
    fn from(err: serde_json::Error) -> Self {
        ExtractionError::ParseError(err.to_string())
    }
}

impl From<candle_core::Error> for ExtractionError {
    fn from(err: candle_core::Error) -> Self {
        ExtractionError::CandleError(err.to_string())
    }
}

/// Extracted article result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExtractedArticle {
    pub url: String,
    pub title: Option<String>,
    pub date: Option<String>,
    pub content: String,
    pub quality_score: f32,
    pub method: String,
    pub xpath: Option<String>,
}

/// Batch extraction result
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BatchExtractionResult {
    pub articles: Vec<ExtractedArticle>,
}