//! Article Extractor - RL-based HTML article extraction library
//!
//! This library provides functionality for extracting article content from HTML
//! using reinforcement learning with fallback to heuristic-based extraction.

pub mod config;
pub mod text_utils;
pub mod html_parser;
pub mod site_profile;
pub mod baseline_extractor;
pub mod models;
pub mod agent;
pub mod environment;
pub mod replay_buffer;
pub mod reward;
pub mod curriculum;
pub mod training;
pub mod hyperparameter;

// Re-exports
pub use config::Config;
pub use site_profile::{SiteProfile, SiteProfileMemory};
pub use baseline_extractor::BaselineExtractor;
pub use agent::DQNAgent;
pub use environment::ArticleExtractionEnvironment;
pub use training::{train_standard, train_with_improvements, TrainingMetrics};
pub use hyperparameter::{HyperparameterSearch, GridSearchConfig};
pub mod checkpoint;
pub use checkpoint::{Checkpoint, CheckpointManager};

use thiserror::Error;

#[cfg(feature = "onnx")]
pub mod onnx_export;

#[cfg(feature = "onnx")]
pub use onnx_export::OnnxModelExporter;

/// Result type for article extraction operations
pub type Result<T> = std::result::Result<T, ExtractionError>;

/// Errors that can occur during article extraction
#[derive(Error, Debug)]
pub enum ExtractionError {
    #[error("Failed to parse HTML: {0}")]
    HtmlParseError(String),

    #[error("Failed to extract article: {0}")]
    ExtractionError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Training error: {0}")]
    TrainingError(String),
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
