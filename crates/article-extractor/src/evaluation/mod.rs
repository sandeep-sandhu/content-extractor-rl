// ============================================================================
// FILE: crates/article-extractor/src/evaluation/mod.rs
// ============================================================================

pub mod ground_truth;
pub mod algorithm_comparison;

pub use ground_truth::{GroundTruthData, GroundTruthEvaluator, EvaluationMetrics};
pub use algorithm_comparison::{AlgorithmComparator, ComparisonReport, AlgorithmResult, RunResult};